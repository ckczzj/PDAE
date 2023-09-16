import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import dataset as dataset_module
from diffusion.gaussian_diffusion import GaussianDiffusion
import model.representation_learning.encoder as encoder_module
import model.representation_learning.decoder as decoder_module
from utils.utils import load_yaml, move_to_cuda, set_worker_seed_builder

from base_sampler import BaseSampler

class Sampler(BaseSampler):
    def __init__(self, args):
        super().__init__(args)
        print('rank{}: sampler initialized.'.format(self.global_rank))

    def _build_dataloader(self):
        dataset_config = self.config["dataset_config"]
        dataset = getattr(dataset_module, dataset_config["dataset_name"], None)(dataset_config)
        total_num = self.config["total_num"]
        subset = Subset(
            dataset,
            np.random.choice(list(range(len(dataset))), total_num, True)
        )


        dataset_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=subset,
            num_replicas=self.global_world_size,
            rank=self.global_rank,
            shuffle=False,
            drop_last=False,
        )

        self.dataloader = DataLoader(
            dataset=subset,
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
        predicted_posterior_mean_gap_list = []
        autoencoder_posterior_mean_gap_list = []
        with torch.inference_mode():
            for batch in self.dataloader:
                x_0 = move_to_cuda(batch["net_input"]["x_0"])
                predicted_posterior_mean_gap, autoencoder_posterior_mean_gap = self.gaussian_diffusion.representation_learning_gap_measure(self.encoder, self.decoder, x_0)
                predicted_posterior_mean_gap_list.append(predicted_posterior_mean_gap)
                autoencoder_posterior_mean_gap_list.append(autoencoder_posterior_mean_gap)

            gather_predicted_posterior_mean_gap_list = self.gather_data(predicted_posterior_mean_gap_list)
            gather_autoencoder_posterior_mean_gap_list = self.gather_data(autoencoder_posterior_mean_gap_list)

            if self.global_rank == 0:
                predicted_posterior_mean_gap_list = []
                autoencoder_posterior_mean_gap_list = []
                for data in gather_predicted_posterior_mean_gap_list:
                    predicted_posterior_mean_gap_list.extend(data)
                for data in gather_autoencoder_posterior_mean_gap_list:
                    autoencoder_posterior_mean_gap_list.extend(data)


                average_predicted_posterior_mean_gap = [np.mean([item[i] for item in predicted_posterior_mean_gap_list]) for i in range(1000)]
                average_autoencoder_posterior_mean_gap = [np.mean([item[i] for item in autoencoder_posterior_mean_gap_list]) for i in range(1000)]

                plt.plot(list(reversed(range(1000))), average_predicted_posterior_mean_gap)
                plt.plot(list(reversed(range(1000))), average_autoencoder_posterior_mean_gap)
                plt.legend(("Pre-trained DPM", "PDAE"))

                plt.xlabel("timestep")
                plt.ylabel("MSE")
                plt.savefig("./gap_measure_result.png", dpi=600)
                plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.config = {
        "config_path": "./trained-models/autoencoder/ffhq128/config.yml",
        "checkpoint_path": "./trained-models/autoencoder/ffhq128/checkpoint.pt",
        "trained_ddpm_config_path": "./pre-trained-dpms/ffhq128/config.yml",

        "dataset_config": {
            "dataset_name": "FFHQ",
            "data_path": "./data/ffhq",
            "image_channel": 3,
            "image_size": 128,
            "augmentation": False,
        },

        "batch_size": 100,
        "total_num": 1000,
        "num_workers": 0,
    }

    runner = Sampler(args)
    runner.start()