import sys
sys.path.append("../")

import copy
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from model.diffusion import GaussianDiffusion
import model.representation.decoder as decoder_module
import model.representation.latent_denoise_fn as latent_denoise_fn_module
from utils import load_yaml, save_image, init_process

from base_sampler import BaseSampler

class Sampler(BaseSampler):
    def __init__(self, config, distributed_meta_info):
        super().__init__(config=config, distributed_meta_info=distributed_meta_info)
        print('rank{}: sampler initialized.'.format(self.rank))

    def _build_dataloader(self):
        pass

    def _build_model(self):
        config_path = self.config["config_path"]
        model_config = load_yaml(config_path)
        self.gaussian_diffusion = GaussianDiffusion(model_config["diffusion_config"], device=self.device)
        trained_ddpm_config = load_yaml(model_config["trained_ddpm_config"])
        decoder = getattr(decoder_module, model_config["decoder_config"]["model"], None)(latent_dim=model_config["decoder_config"]["latent_dim"], **trained_ddpm_config["denoise_fn_config"])
        self.decoder = DistributedDataParallel(copy.deepcopy(decoder).cuda(), device_ids=[self.device])
        del decoder
        checkpoint_path = self.config["checkpoint_path"]
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.decoder.module.load_state_dict(checkpoint['ema_decoder'])

        self.decoder.requires_grad_(False)
        self.decoder.eval()

        latent_denoise_fn_config_path = self.config["latent_denoise_fn_config_path"]
        latent_denoise_fn_config = load_yaml(latent_denoise_fn_config_path)
        latent_denoise_fn = getattr(latent_denoise_fn_module, latent_denoise_fn_config["latent_denoise_fn_config"]["model"], None)(**latent_denoise_fn_config["latent_denoise_fn_config"])
        self.latent_denoise_fn = DistributedDataParallel(copy.deepcopy(latent_denoise_fn).cuda(), device_ids=[self.device])
        del latent_denoise_fn

        latent_denoise_fn_checkpoint_path = self.config["latent_denoise_fn_checkpoint_path"]
        latent_denoise_fn_checkpoint = torch.load(latent_denoise_fn_checkpoint_path, map_location=torch.device('cpu'))
        self.latent_denoise_fn.module.load_state_dict(latent_denoise_fn_checkpoint['ema_latent_denoise_fn'])

        self.latent_denoise_fn.requires_grad_(False)
        self.latent_denoise_fn.eval()

        inferred_latents = torch.load(self.config["inferred_latents_path"], map_location=torch.device('cpu'))
        self.latents_mean = inferred_latents["mean"].to(self.device)
        self.latents_std = inferred_latents["std"].to(self.device)

    def start(self):
        num = self.dispatch_num_samples_for_process(self.config["batch_size"], self.world_size, self.rank)

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
        if self.rank == 0:
            images = []
            for data in gather_data:
                images.extend(data)
            images = np.asarray(images, dtype=np.uint8)
            save_image(images, "./unconditional_sample_result.png")


def run(rank, args, distributed_meta_info):
    distributed_meta_info["rank"] = rank
    init_process(
        init_method=distributed_meta_info["init_method"],
        rank=distributed_meta_info["rank"],
        world_size=distributed_meta_info["world_size"]
    )

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    sampler = Sampler(
        config=args.config,
        distributed_meta_info=distributed_meta_info
    )
    sampler.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--world_size', type=str, required=True)
    parser.add_argument('--master_addr', type=str, default="127.0.0.1")
    parser.add_argument('--master_port', type=str, default="6666")

    args = parser.parse_args()

    args.config = {
        "config_path": "../trained-models/autoencoder/celeba64/config.yml",
        "checkpoint_path": "../trained-models/autoencoder/celeba64/checkpoint.pt",

        "inferred_latents_path": "../trained-models/latents/celeba64.pt",

        "latent_denoise_fn_config_path": "../trained-models/latent_denoise_fn/celeba64/config.yml",
        "latent_denoise_fn_checkpoint_path": "../trained-models/latent_denoise_fn/celeba64/checkpoint.pt",

        "latent_ddim_style": f'ddim100',
        "decoder_ddim_style": f'ddim100',

        "image_channel": 3,
        "image_size": 64,

        "batch_size": 36,
    }

    world_size = int(args.world_size)
    init_method = "tcp://{}:{}".format(args.master_addr, args.master_port)

    distributed_meta_info = {
        "world_size": world_size,
        "master_addr": args.master_addr,
        "init_method": init_method,
        # rank will be added in spawned processes
    }

    mp.spawn(
        fn=run,
        args=(args, distributed_meta_info),
        nprocs=world_size,
        join=True,
        daemon=False
    )

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 unconditional_sample.py --world_size 4