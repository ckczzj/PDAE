import argparse

import torch

from diffusion.gaussian_diffusion import GaussianDiffusion
from model.unet import UNet
from utils.utils import load_yaml, move_to_cuda, save_image

from sampler.base_sampler import BaseSampler

class Sampler(BaseSampler):
    def __init__(self, args):
        super().__init__(args)
        print('rank{}: sampler initialized.'.format(self.global_rank))

    def _build_dataloader(self):
        pass

    def _build_model(self):
        self.gaussian_diffusion = GaussianDiffusion(self.config["diffusion_config"], device=self.device)

        config_path = self.config["config_path"]
        checkpoint_path = self.config["checkpoint_path"]
        model_config = load_yaml(config_path)
        self.denoise_fn = UNet(**model_config["denoise_fn_config"])
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.denoise_fn.load_state_dict(checkpoint["ema_denoise_fn"])
        self.denoise_fn = self.denoise_fn.cuda()
        self.denoise_fn.eval()

    def start(self):
        image_channel = self.config["image_channel"]
        image_size = self.config["image_size"]

        with torch.inference_mode():
            x_T = move_to_cuda(torch.randn(9, image_channel, image_size, image_size))
            samples = self.gaussian_diffusion.test_pretrained_dpms(f'ddim100', self.denoise_fn, x_T)

            samples = samples.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
            samples = samples.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()

            save_image(samples, "./test_dpms_result.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.config = {
        "diffusion_config" : {
            "timesteps": 1000,
            "betas_type": "linear",
        },

        "config_path": "./pre-trained-dpms/ffhq128/config.yml",
        "checkpoint_path": "./pre-trained-dpms/ffhq128/checkpoint.pt",

        "image_channel": 3,
        "image_size": 128,
    }

    runner = Sampler(args)
    runner.start()
