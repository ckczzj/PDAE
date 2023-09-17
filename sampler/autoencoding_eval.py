import copy
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import lpips

import dataset as dataset_module
from diffusion.gaussian_diffusion import GaussianDiffusion
import model.representation_learning.encoder as encoder_module
import model.representation_learning.decoder as decoder_module
from utils.utils import load_yaml, move_to_cuda, calculate_ssim, calculate_lpips, calculate_mse, set_worker_seed_builder

from sampler.base_sampler import BaseSampler

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

    def start(self):
        lpips_fn = lpips.LPIPS(net='alex').to(self.device)

        ssim_score_list = []
        lpips_score_list = []
        mse_score_list = []

        with torch.inference_mode():
            for batch in self.dataloader:
                x_0 = move_to_cuda(batch["x_0"])

                # inferred x_T
                reconstruction = self.gaussian_diffusion.representation_learning_autoencoding(f'ddim1000', f'ddim100', self.encoder, self.decoder, x_0)

                # random x_T
                # reconstruction = self.gaussian_diffusion.representation_learning_ddim_sample(f'ddim100', self.encoder, self.decoder, x_0, torch.randn_like(x_0))

                norm_x_0 = (x_0 + 1.) / 2.
                norm_reconstruction = (reconstruction + 1.) / 2.

                ssim_score = calculate_ssim(norm_x_0, norm_reconstruction)
                lpips_score = calculate_lpips(x_0, reconstruction, lpips_fn)
                mse_score = calculate_mse(norm_x_0, norm_reconstruction)

                ssim_score_list.extend(ssim_score.tolist())
                lpips_score_list.extend(lpips_score.tolist())
                mse_score_list.extend(mse_score.tolist())

                gather_ssim_score_list = self.gather_data(ssim_score_list)
                gather_lpips_score_list = self.gather_data(lpips_score_list)
                gather_mse_score_list = self.gather_data(mse_score_list)

                if self.global_rank == 0:
                    all_ssim_score_list = []
                    all_lpips_score_list = []
                    all_mse_score_list = []
                    for data in gather_ssim_score_list:
                        all_ssim_score_list.extend(data)
                    for data in gather_lpips_score_list:
                        all_lpips_score_list.extend(data)
                    for data in gather_mse_score_list:
                        all_mse_score_list.extend(data)

                    print(len(all_ssim_score_list), len(all_lpips_score_list), len(all_mse_score_list))
                    print(np.mean(all_ssim_score_list), np.mean(all_lpips_score_list), np.mean(all_mse_score_list))


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
