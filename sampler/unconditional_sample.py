import copy
import argparse
import numpy as np
import torch

from diffusion.gaussian_diffusion import GaussianDiffusion
import model.representation_learning.decoder as decoder_module
import model.representation_learning.latent_denoise_fn as latent_denoise_fn_module
from utils.utils import load_yaml, save_image

from base_sampler import BaseSampler

class Sampler(BaseSampler):
    def __init__(self, args):
        super().__init__(args)
        print('rank{}: sampler initialized.'.format(self.global_rank))

    def _build_dataloader(self):
        pass

    def _build_model(self):
        config_path = self.config["config_path"]
        model_config = load_yaml(config_path)
        self.gaussian_diffusion = GaussianDiffusion(model_config["diffusion_config"], device=self.device)
        trained_ddpm_config = load_yaml(self.config["trained_ddpm_config_path"])
        decoder = getattr(decoder_module, model_config["decoder_config"]["model"], None)(latent_dim=model_config["decoder_config"]["latent_dim"], **trained_ddpm_config["denoise_fn_config"])
        self.decoder = copy.deepcopy(decoder).cuda()
        del decoder
        checkpoint_path = self.config["checkpoint_path"]
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.decoder.load_state_dict(checkpoint['ema_decoder'])

        self.decoder.requires_grad_(False)
        self.decoder.eval()

        latent_denoise_fn_config_path = self.config["latent_denoise_fn_config_path"]
        latent_denoise_fn_config = load_yaml(latent_denoise_fn_config_path)
        latent_denoise_fn = getattr(latent_denoise_fn_module, latent_denoise_fn_config["latent_denoise_fn_config"]["model"], None)(**latent_denoise_fn_config["latent_denoise_fn_config"])
        self.latent_denoise_fn = copy.deepcopy(latent_denoise_fn).cuda()
        del latent_denoise_fn

        latent_denoise_fn_checkpoint_path = self.config["latent_denoise_fn_checkpoint_path"]
        latent_denoise_fn_checkpoint = torch.load(latent_denoise_fn_checkpoint_path, map_location=torch.device('cpu'))
        self.latent_denoise_fn.load_state_dict(latent_denoise_fn_checkpoint['ema_latent_denoise_fn'])

        self.latent_denoise_fn.requires_grad_(False)
        self.latent_denoise_fn.eval()

        inferred_latents = torch.load(self.config["inferred_latents_path"], map_location=torch.device('cpu'))
        self.latents_mean = inferred_latents["mean"].to(self.device)
        self.latents_std = inferred_latents["std"].to(self.device)

    def start(self):
        num = self.dispatch_num_samples_for_process(self.config["total_num"], self.global_world_size, self.global_rank)

        images = self.gaussian_diffusion.latent_diffusion_sample(
            latent_ddim_style=self.config["latent_ddim_style"],
            decoder_ddim_style=self.config["latent_ddim_style"],
            latent_denoise_fn=self.latent_denoise_fn,
            decoder=self.decoder,
            x_T=torch.randn(num, self.config["image_channel"],self.config["image_size"],self.config["image_size"]).cuda(),
            latents_mean=self.latents_mean,
            latents_std=self.latents_std,
        )

        images = images.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
        images = images.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()

        gather_data = self.gather_data(images)
        if self.global_rank == 0:
            images = []
            for data in gather_data:
                images.extend(data)
            images = np.asarray(images, dtype=np.uint8)
            save_image(images, "./unconditional_sample_result.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.config = {
        "config_path": "./trained-models/autoencoder/celeba64/config.yml",
        "checkpoint_path": "./trained-models/autoencoder/celeba64/checkpoint.pt",
        "trained_ddpm_config_path": "./pre-trained-dpms/celeba64/config.yml",

        "inferred_latents_path": "./trained-models/latents/celeba64.pt",

        "latent_denoise_fn_config_path": "./trained-models/latent_denoise_fn/celeba64/config.yml",
        "latent_denoise_fn_checkpoint_path": "./trained-models/latent_denoise_fn/celeba64/checkpoint.pt",

        "latent_ddim_style": f'ddim100',
        "decoder_ddim_style": f'ddim100',

        "image_channel": 3,
        "image_size": 64,

        "total_num": 36,
    }

    runner = Sampler(args)
    runner.start()
