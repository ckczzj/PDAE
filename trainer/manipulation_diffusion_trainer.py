import copy
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam

import model.representation.encoder as encoder_module
import model.representation.decoder as decoder_module
from model.diffusion import GaussianDiffusion

from utils import move_to_cuda, save_image, load_yaml
from base_trainer import BaseTrainer


class ManipulationDiffusionTrainer(BaseTrainer):
    def __init__(self, config, run_path, distributed_meta_info):
        super().__init__(config=config, run_path=run_path, distributed_meta_info=distributed_meta_info)
        print('rank{}: trainer initialized.'.format(self.rank))

    def _build_model(self):
        trained_representation_learning_config = load_yaml(self.config["trained_representation_learning_config"])

        self.gaussian_diffusion = GaussianDiffusion(trained_representation_learning_config["diffusion_config"], device=self.device)

        classifier = nn.Linear(512, 40)
        self.classifier = DistributedDataParallel(copy.deepcopy(classifier).cuda(), device_ids=[self.device])
        self.ema_classifier = DistributedDataParallel(copy.deepcopy(classifier).cuda(), device_ids=[self.device])
        del classifier

        self.ema_classifier.eval()
        self.ema_classifier.requires_grad_(False)

        encoder = getattr(encoder_module, trained_representation_learning_config["encoder_config"]["model"], None)(**trained_representation_learning_config["encoder_config"])
        self.encoder = DistributedDataParallel(copy.deepcopy(encoder).cuda(), device_ids=[self.device])
        del encoder
        self.load_trained_encoder(self.config["trained_representation_learning_checkpoint"])

        trained_ddpm_config = load_yaml(trained_representation_learning_config["trained_ddpm_config"])
        decoder = getattr(decoder_module, trained_representation_learning_config["decoder_config"]["model"], None)(latent_dim = trained_representation_learning_config["decoder_config"]["latent_dim"], **trained_ddpm_config["denoise_fn_config"])
        self.decoder = DistributedDataParallel(copy.deepcopy(decoder).cuda(), device_ids=[self.device])
        del decoder
        self.load_trained_decoder(self.config["trained_representation_learning_checkpoint"])

        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.decoder.eval()
        self.decoder.requires_grad_(False)

        # load latents stat
        self.load_inferred_latents(self.config["inferred_latents"])
        self.latents_mean = self.latents['mean'].to(self.device)
        self.latents_std = self.latents['std'].to(self.device)

        self.enable_amp = self.config["optimizer_config"]["enable_amp"]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)


    def _build_optimizer(self):
        optimizer_config = self.config["optimizer_config"]

        self.optimizer = Adam(
            [
                {"params": self.classifier.parameters()},
            ],
            lr = float(optimizer_config["lr"]),
            betas = eval(optimizer_config["adam_betas"]),
            eps = float(optimizer_config["adam_eps"]),
            weight_decay= float(optimizer_config["weight_decay"]),
        )

    def train(self):
        acc_bce_loss = 0
        acc_final_loss = 0
        time_meter = defaultdict(float)

        display_steps = 100
        while True:
            start_time_top = time.time_ns()

            self.classifier.train()
            self.optimizer.zero_grad()

            # to solve small batch size for large data
            num_iterations = self.config["runner_config"]["num_iterations"]

            for _ in range(num_iterations):

                start_time = time.time_ns()
                batch = next(self.train_dataloader_infinite_cycle)
                time_meter['load data'] += (time.time_ns() - start_time) / 1e9

                with torch.cuda.amp.autocast(enabled=self.enable_amp):
                    start_time = time.time_ns()
                    output = self.gaussian_diffusion.manipulation_train_one_batch(
                        classifier=self.classifier,
                        encoder=self.encoder,
                        x_0=move_to_cuda(batch["net_input"]["x_0"]),
                        label=move_to_cuda(batch["net_input"]["label"]),
                        latents_mean=move_to_cuda(self.latents_mean),
                        latents_std=move_to_cuda(self.latents_std),
                    )
                    time_meter['forward'] += (time.time_ns() - start_time) / 1e9

                    bce_loss = output['bce_loss'] / num_iterations
                    final_loss = bce_loss

                    acc_bce_loss += bce_loss.item()
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
                info = 'rank{}: step = {}, bce = {:.5f}, final = {:.5f}, lr = {:.6f}'.format(
                    self.rank, self.step,
                    acc_bce_loss / display_steps,
                    acc_final_loss / display_steps,
                    self.optimizer.defaults["lr"])
                print('{} '.format(info), end=' - ')
                for k, v in time_meter.items():
                    print('{}: {:.2f} secs'.format(k, v), end=', ')
                print()

                data = {'acc_bce_loss': acc_bce_loss, 'acc_final_loss': acc_final_loss}
                gather_data = self.gather_data(data)
                if self.rank == 0:
                    self.writer.add_scalar("bce_loss", float(np.mean([data["acc_bce_loss"] for data in gather_data])) / display_steps, self.step)
                    self.writer.add_scalar("final_loss", float(np.mean([data["acc_final_loss"] for data in gather_data])) / display_steps, self.step)
                    self.writer.add_scalar("learning_rate", self.optimizer.defaults["lr"], self.step)

                acc_bce_loss = 0
                acc_final_loss = 0
                time_meter.clear()

            if self.rank == 0 and self.step % self.config["runner_config"]["save_latest_every_steps"] == 0:
                self.save(os.path.join(os.path.join(self.run_path, 'checkpoints'), 'latest.pt'))
            if self.rank == 0 and self.step % self.config["runner_config"]["save_checkpoint_every_steps"] == 0:
                self.save(os.path.join(os.path.join(self.run_path, 'checkpoints'), 'save-{}k.pt'.format(self.step // 1000)))
            if self.step % self.config["runner_config"]["evaluate_every_steps"] == 0:
                self.eval()

    def eval(self):
        with torch.no_grad():
            torch.distributed.barrier()
            # ensure to generate different samples
            self.eval_sampler.set_epoch(self.step)

            for batch in self.eval_dataloader:
                x_0 = move_to_cuda(batch["net_input"]["x_0"])
                inferred_x_T = self.gaussian_diffusion.representation_learning_ddim_encode(f'ddim500', self.encoder, self.decoder, x_0)
                images = self.gaussian_diffusion.manipulation_sample(
                    classifier_weight=self.ema_classifier.module.weight,
                    encoder=self.encoder,
                    decoder=self.decoder,
                    x_0=x_0,
                    inferred_x_T=inferred_x_T,
                    latents_mean=move_to_cuda(self.latents_mean),
                    latents_std=move_to_cuda(self.latents_std),
                    class_id=31,
                    scale=0.3,
                    ddim_style=f'ddim200',
                )
                images = images.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
                images = images.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                data = {'images': images.tolist()}
                data.update({'gts': batch['gts'].tolist()})
                gather_data = self.gather_data(data)
                break

            if self.rank == 0:
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
        self.classifier.eval()
        self.ema_classifier.eval()

        ema_classifier_parameter = dict(self.ema_classifier.named_parameters())
        classifier_parameter = dict(self.classifier.named_parameters())

        for k in ema_classifier_parameter.keys():
            if classifier_parameter[k].requires_grad:
                ema_classifier_parameter[k].data.mul_(decay).add_(classifier_parameter[k].data, alpha=1.0 - decay)

        # batchnorm layer has running stat buffer
        # dict(model.named_buffers())

    def save(self, path):
        data = {
            'step': self.step,
            'encoder': self.encoder.module.state_dict(),
            'decoder': self.decoder.module.state_dict(),
            'classifier': self.classifier.module.state_dict(),
            'ema_classifier': self.ema_classifier.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, path)

        print('rank{}: step, model, optimizer and scaler saved to {}(step {}k).'.format(self.rank, path, self.step // 1000))

    def load(self, path):
        data = torch.load(path, map_location=torch.device('cpu'))

        self.step = data['step']
        self.encoder.module.load_state_dict(data['encoder'])
        self.decoder.module.load_state_dict(data['decoder'])
        self.classifier.module.load_state_dict(data['classifier'])
        self.ema_classifier.module.load_state_dict(data['ema_classifier'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.scaler.load_state_dict(data['scaler'])

        print('rank{}: step, model, optimizer and scaler restored from {}(step {}k).'.format(self.rank, path, self.step // 1000))

    def load_trained_encoder(self, path):
        data = torch.load(path, map_location=torch.device('cpu'))
        self.encoder.module.load_state_dict(data['ema_encoder'])

    def load_trained_decoder(self, path):
        data = torch.load(path, map_location=torch.device('cpu'))
        self.decoder.module.load_state_dict(data['ema_decoder'])

    def load_inferred_latents(self, path):
        self.latents = torch.load(path, map_location=torch.device('cpu'))
