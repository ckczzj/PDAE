import argparse
import copy
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam

import model.representation_learning.encoder as encoder_module
import model.representation_learning.decoder as decoder_module
from diffusion.gaussian_diffusion import GaussianDiffusion

from utils.utils import move_to_cuda, save_image, load_yaml
from trainer.base_trainer import BaseTrainer


class RepresentationLearningTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        print('rank{}: trainer initialized.'.format(self.global_rank))

    def _build_model(self):
        self.gaussian_diffusion = GaussianDiffusion(self.config["diffusion_config"], device=self.device)

        encoder = getattr(encoder_module, self.config["encoder_config"]["model"], None)(**self.config["encoder_config"])
        self.encoder = DistributedDataParallel(copy.deepcopy(encoder).cuda(), device_ids=[self.device])
        self.encoder_without_ddp = self.encoder.module
        self.ema_encoder = copy.deepcopy(encoder).cuda()
        del encoder

        self.ema_encoder.eval()
        self.ema_encoder.requires_grad_(False)

        trained_ddpm_config = load_yaml(self.config["trained_ddpm_config"])
        decoder = getattr(decoder_module, self.config["decoder_config"]["model"], None)(latent_dim = self.config["decoder_config"]["latent_dim"], **trained_ddpm_config["denoise_fn_config"])
        self.decoder = DistributedDataParallel(copy.deepcopy(decoder).cuda(), device_ids=[self.device])
        self.decoder_without_ddp = self.decoder.module
        self.ema_decoder = copy.deepcopy(decoder).cuda()
        del decoder
        self.load_trained_ddpm(self.config["trained_ddpm_checkpoint"])

        self.ema_decoder.eval()
        self.ema_decoder.requires_grad_(False)

        self.enable_amp = self.config["optimizer_config"]["enable_amp"]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)

        if self.config["runner_config"]["compile"]:
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)

    def _build_optimizer(self):
        optimizer_config = self.config["optimizer_config"]

        self.optimizer = Adam(
            [
                {"params": self.encoder_without_ddp.parameters()},
                {"params": self.decoder_without_ddp.label_emb.parameters()},
                {"params": self.decoder_without_ddp.shift_middle_block.parameters()},
                {"params": self.decoder_without_ddp.shift_output_blocks.parameters()},
                {"params": self.decoder_without_ddp.shift_out.parameters()}
            ],
            lr = float(optimizer_config["lr"]),
            betas = eval(optimizer_config["adam_betas"]),
            eps = float(optimizer_config["adam_eps"]),
            weight_decay= float(optimizer_config["weight_decay"]),
        )

    def train(self):
        acc_prediction_loss = 0
        acc_final_loss = 0
        time_meter = defaultdict(float)

        display_steps = self.config["runner_config"]["display_steps"]
        while True:
            start_time_top = time.time_ns()

            self.encoder.train()
            self.decoder.module.set_train_mode()
            self.optimizer.zero_grad()

            # to solve small batch size for large data
            num_iterations = self.config["runner_config"]["num_iterations"]

            for _ in range(num_iterations):

                start_time = time.time_ns()
                batch = next(self.train_dataloader_infinite_cycle)
                time_meter['load data'] += (time.time_ns() - start_time) / 1e9

                with torch.cuda.amp.autocast(enabled=self.enable_amp):
                    start_time = time.time_ns()
                    output = self.gaussian_diffusion.representation_learning_train_one_batch(
                        encoder=self.encoder,
                        decoder=self.decoder,
                        x_0=move_to_cuda(batch["net_input"]["x_0"])
                    )
                    # torch.distributed.barrier()   CUDA operations are executed asynchronously, the timing will be accumulated in the next synchronizing operation
                    time_meter['forward'] += (time.time_ns() - start_time) / 1e9

                    prediction_loss = output['prediction_loss'] / num_iterations
                    final_loss = prediction_loss

                    acc_prediction_loss += prediction_loss.item()
                    acc_final_loss += final_loss.item()

                start_time = time.time_ns()
                self.scaler.scale(final_loss).backward()
                time_meter['backward'] += (time.time_ns() - start_time) / 1e9

            start_time = time.time_ns()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            time_meter['param update'] += (time.time_ns() - start_time) / 1e9

            self.step += 1

            if self.step % self.config["runner_config"]["ema_every"] == 0:
                start_time = time.time_ns()
                self.accumulate(self.config["runner_config"]["ema_decay"])
                time_meter['accumulate'] += (time.time_ns() - start_time) / 1e9


            time_meter['step'] += (time.time_ns() - start_time_top) / 1e9

            if self.step % display_steps == 0:
                info = 'rank{}: step = {}, pred = {:.5f}, final = {:.5f}, lr = {:.6f}'.format(
                    self.global_rank, self.step,
                    acc_prediction_loss / display_steps,
                    acc_final_loss / display_steps,
                    self.optimizer.defaults["lr"])
                print('{} '.format(info), end=' - ')
                for k, v in time_meter.items():
                    print('{}: {:.2f} secs'.format(k, v), end=', ')
                print()

                data = {'acc_prediction_loss': acc_prediction_loss, 'acc_final_loss': acc_final_loss}
                gather_data = self.gather_data(data)
                if self.global_rank == 0:
                    self.writer.add_scalar("prediction_loss", float(np.mean([data["acc_prediction_loss"] for data in gather_data])) / display_steps, self.step)
                    self.writer.add_scalar("final_loss", float(np.mean([data["acc_final_loss"] for data in gather_data])) / display_steps, self.step)
                    self.writer.add_scalar("learning_rate", self.optimizer.defaults["lr"], self.step)

                acc_prediction_loss = 0
                acc_final_loss = 0
                time_meter.clear()

            if self.global_rank == 0 and self.step % self.config["runner_config"]["save_latest_every_steps"] == 0:
                self.save(os.path.join(os.path.join(self.run_path, 'checkpoints'), 'latest.pt'))
            if self.global_rank == 0 and self.step % self.config["runner_config"]["save_checkpoint_every_steps"] == 0:
                self.save(os.path.join(os.path.join(self.run_path, 'checkpoints'), 'save-{}k.pt'.format(self.step // 1000)))
            if self.step % self.config["runner_config"]["evaluate_every_steps"] == 0:
                self.eval()

    def eval(self):
        with torch.no_grad():
            torch.distributed.barrier()

            # ensure to generate different samples
            self.eval_sampler.set_epoch(self.step)

            for batch in self.eval_dataloader:
                images = self.gaussian_diffusion.representation_learning_ddim_sample(
                    ddim_style=f'ddim100',
                    encoder=self.ema_encoder,
                    decoder=self.ema_decoder,
                    x_0=move_to_cuda(batch["net_input"]["x_0"]),
                    x_T=move_to_cuda(batch["net_input"]["x_T"]),
                )
                images = images.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
                images = images.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                data = {'images': images.tolist()}
                data.update({'gts': batch['gts'].tolist()})
                gather_data = self.gather_data(data)
                break

            if self.global_rank == 0:
                images = []
                gts = []
                for data in gather_data:
                    images.extend(data["images"])
                    gts.extend(data['gts'])
                images = np.asarray(images, dtype=np.uint8)
                gts = np.asarray(gts, dtype=np.uint8)
                figure = save_image(images, os.path.join(self.run_path, 'samples', "sample{}k.png".format(self.step // 1000)),gts=gts)
                # only writer of rank0 is None
                self.writer.add_figure("result", figure, self.step)

    def accumulate(self, decay):
        self.encoder.eval()
        self.ema_encoder.eval()

        ema_encoder_parameter = dict(self.ema_encoder.named_parameters())
        encoder_parameter = dict(self.encoder_without_ddp.named_parameters())

        for k in ema_encoder_parameter.keys():
            if encoder_parameter[k].requires_grad:
                ema_encoder_parameter[k].data.mul_(decay).add_(encoder_parameter[k].data, alpha=1.0 - decay)


        self.decoder.eval()
        self.ema_decoder.eval()

        ema_decoder_parameter = dict(self.ema_decoder.named_parameters())
        decoder_parameter = dict(self.decoder_without_ddp.named_parameters())

        for k in ema_decoder_parameter.keys():
            if decoder_parameter[k].requires_grad:
                ema_decoder_parameter[k].data.mul_(decay).add_(decoder_parameter[k].data, alpha=1.0 - decay)

    def save(self, path):
        data = {
            'step': self.step,
            'encoder': self.encoder_without_ddp.state_dict(),
            'ema_encoder': self.ema_encoder.state_dict(),
            'decoder': self.decoder_without_ddp.state_dict(),
            'ema_decoder': self.ema_decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, path)

        print('rank{}: step, model, optimizer and scaler saved to {}(step {}k).'.format(self.global_rank, path, self.step // 1000))

    def load(self, path):
        data = torch.load(path, map_location=torch.device('cpu'))

        self.step = data['step']
        self.encoder_without_ddp.load_state_dict(data['encoder'])
        self.ema_encoder.load_state_dict(data['ema_encoder'])
        self.decoder_without_ddp.load_state_dict(data['decoder'])
        self.ema_decoder.load_state_dict(data['ema_decoder'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.scaler.load_state_dict(data['scaler'])

        print('rank{}: step, model, optimizer and scaler restored from {}(step {}k).'.format(self.global_rank, path, self.step // 1000))

    def load_trained_ddpm(self, path):
        data = torch.load(path, map_location=torch.device('cpu'))
        self.decoder_without_ddp.load_state_dict(data['ema_denoise_fn'], strict=False)
        self.ema_decoder.load_state_dict(data['ema_denoise_fn'], strict=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--run_path', type=str, required=True)
    parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')

    args = parser.parse_args()
    runner = RepresentationLearningTrainer(args)
    runner.train()