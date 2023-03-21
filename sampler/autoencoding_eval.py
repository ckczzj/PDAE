import sys
sys.path.append("../")

import copy
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import lpips

import dataset as dataset_module
from model.diffusion import GaussianDiffusion
import model.representation.encoder as encoder_module
import model.representation.decoder as decoder_module
from utils import load_yaml, move_to_cuda, calculate_ssim, calculate_lpips, calculate_mse, set_worker_seed_builder, init_process

from base_sampler import BaseSampler

class Sampler(BaseSampler):
    def __init__(self, config, distributed_meta_info):
        super().__init__(config=config, distributed_meta_info=distributed_meta_info)
        print('rank{}: runner initialized.'.format(self.rank))

    def _build_dataloader(self):
        dataset_name = self.config["dataset_name"]
        data_path = self.config["data_path"]
        image_size = self.config["image_size"]
        image_channel = self.config["image_channel"]
        augmentation = self.config["augmentation"]
        dataset = getattr(dataset_module, dataset_name, None)({"data_path": data_path, "image_size": image_size, "image_channel": image_channel, "augmentation": augmentation})

        # dispatch batch_size
        global_batch_size = self.config["batch_size"]
        local_batch_size = global_batch_size // self.world_size
        assert local_batch_size > 0
        num_workers = self.config["num_workers"]

        dataset_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False
        )

        self.dataloader = DataLoader(
            dataset=dataset,
            sampler=dataset_sampler,
            pin_memory=False,
            collate_fn=dataset.collate_fn,
            worker_init_fn=set_worker_seed_builder(self.rank),
            batch_size=local_batch_size,
            num_workers=num_workers
        )

    def _build_model(self):
        config_path = self.config["config_path"]
        model_config = load_yaml(config_path)
        self.gaussian_diffusion = GaussianDiffusion(model_config["diffusion_config"], device=self.device)

        encoder = getattr(encoder_module, model_config["encoder_config"]["model"], None)(**model_config["encoder_config"])
        self.encoder = DistributedDataParallel(copy.deepcopy(encoder).cuda(), device_ids=[self.device])
        del encoder
        trained_ddpm_config = load_yaml(model_config["trained_ddpm_config"])
        decoder = getattr(decoder_module, model_config["decoder_config"]["model"], None)(latent_dim=model_config["decoder_config"]["latent_dim"], **trained_ddpm_config["denoise_fn_config"])
        self.decoder = DistributedDataParallel(copy.deepcopy(decoder).cuda(), device_ids=[self.device])
        del decoder

        checkpoint_path = self.config["checkpoint_path"]
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.encoder.module.load_state_dict(checkpoint['ema_encoder'])
        self.decoder.module.load_state_dict(checkpoint['ema_decoder'])

        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.decoder.requires_grad_(False)
        self.decoder.eval()

    def start(self):
        lpips_fn = lpips.LPIPS(net='alex').to(self.device)

        ssim_score_list = []
        lpips_score_list = []
        mse_score_list = []

        with torch.inference_mode():
            for batch in self.dataloader:
                x_0 = move_to_cuda(batch["net_input"]["x_0"])

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

                if self.rank == 0:
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
        "diffusion_config": {
            "timesteps": 1000,
            "betas_type": "linear",
            "linear_beta_start": 0.0001,
            "linear_beta_end": 0.02
        },

        "config_path": "../trained-models/autoencoder/ffhq128/config.yml",
        "checkpoint_path": "../trained-models/autoencoder/ffhq128/checkpoint.pt",

        "dataset_name": "CELEBAHQ",
        "data_path": "../data/celebahq",
        "image_channel": 3,
        "image_size": 128,
        "augmentation": False,

        "batch_size": 120,
        "num_workers": 2,
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

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 autoencoding_eval.py --world_size 4