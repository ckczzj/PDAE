import copy
import argparse
import torch
from torch.utils.data import DataLoader

import dataset as dataset_module
from diffusion.gaussian_diffusion import GaussianDiffusion
import model.representation_learning.encoder as encoder_module
import model.representation_learning.decoder as decoder_module
from utils.utils import load_yaml, move_to_cuda, set_worker_seed_builder

from sampler.base_sampler import BaseSampler
from metric.lpips.lpips_metric import LPIPSMetric
from metric.ssim.ssim_metric import SSIMMetric
from metric.mse.mse_metric import MSEMetric

class Sampler(BaseSampler):
    def __init__(self, args):
        super().__init__(args)
        print('rank{}: sampler initialized.'.format(self.global_rank))

    def _build_dataloader(self):
        dataset_config = self.config["dataset_config"]
        dataset = getattr(dataset_module, dataset_config["dataset_name"], None)(dataset_config)

        dataset_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=self.global_world_size,
            rank=self.global_rank,
            shuffle=False,
            drop_last=False,
        )

        self.dataloader = DataLoader(
            dataset=dataset,
            sampler=dataset_sampler,
            pin_memory=False,
            collate_fn=dataset.collate_fn,
            worker_init_fn=set_worker_seed_builder(self.global_rank),
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            drop_last=False,
        )

    def _build_model(self):
        config_path = self.config["config_path"]
        model_config = load_yaml(config_path)
        self.gaussian_diffusion = GaussianDiffusion(model_config["diffusion_config"], device=self.device)

        encoder = getattr(encoder_module, model_config["encoder_config"]["model"], None)(**model_config["encoder_config"])
        self.encoder = copy.deepcopy(encoder).cuda()
        del encoder
        trained_ddpm_config = load_yaml(self.config["trained_ddpm_config_path"])
        decoder = getattr(decoder_module, model_config["decoder_config"]["model"], None)(latent_dim=model_config["decoder_config"]["latent_dim"], **trained_ddpm_config["denoise_fn_config"])
        self.decoder = copy.deepcopy(decoder).cuda()
        del decoder

        checkpoint_path = self.config["checkpoint_path"]
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.encoder.load_state_dict(checkpoint['ema_encoder'])
        self.decoder.load_state_dict(checkpoint['ema_decoder'])

        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.decoder.eval()
        self.decoder.requires_grad_(False)

        self.ssim_metric = SSIMMetric()
        self.lpips_metric = LPIPSMetric(device=self.device)
        self.mse_metric = MSEMetric()

    def start(self):
        with torch.inference_mode():
            for batch in self.dataloader:
                x_0 = move_to_cuda(batch["x_0"])

                # inferred x_T
                reconstruction = self.gaussian_diffusion.representation_learning_autoencoding(f'ddim1000', f'ddim100', self.encoder, self.decoder, x_0)

                # random x_T
                # reconstruction = self.gaussian_diffusion.representation_learning_ddim_sample(f'ddim100', self.encoder, self.decoder, x_0, torch.randn_like(x_0))

                norm_x_0 = (x_0 + 1.) / 2.
                norm_reconstruction = (reconstruction + 1.) / 2.

                self.ssim_metric.process(norm_x_0, norm_reconstruction)
                self.lpips_metric.process(x_0, reconstruction)
                self.mse_metric.process(norm_x_0, norm_reconstruction)


        ssim_results = self.ssim_metric.all_gather_results(self.global_world_size)
        lpips_results = self.lpips_metric.all_gather_results(self.global_world_size)
        mse_results = self.mse_metric.all_gather_results(self.global_world_size)
        print(len(ssim_results), len(lpips_results), len(mse_results))
        if self.global_rank == 0:
            ssim = self.ssim_metric.compute_metrics(ssim_results)
            lpips = self.lpips_metric.compute_metrics(lpips_results)
            mse = self.mse_metric.compute_metrics(mse_results)
            print("ssim: ", ssim, "lpips: ", lpips, "mse: ", mse)
        torch.distributed.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.config = {
        "diffusion_config": {
            "timesteps": 1000,
            "betas_type": "linear",
        },

        "config_path": "./trained-models/autoencoder/ffhq128/config.yml",
        "checkpoint_path": "./trained-models/autoencoder/ffhq128/checkpoint.pt",
        "trained_ddpm_config_path": "./pre-trained-dpms/ffhq128/config.yml",

        "dataset_config": {
            "dataset_name": "CELEBAHQ",
            "data_path": "./data/celebahq",
            "image_channel": 3,
            "image_size": 128,
            "augmentation": False
        },

        "batch_size": 100,
        "num_workers": 2,
    }

    runner = Sampler(args)
    runner.start()
