import argparse

import torch
from torch.utils.data import DataLoader

import dataset as dataset_module
from diffusion.gaussian_diffusion import GaussianDiffusion
import model.representation_learning.encoder as encoder_module
from utils.utils import load_yaml, move_to_cuda

from sampler.base_sampler import BaseSampler

class Sampler(BaseSampler):
    def __init__(self, args):
        super().__init__(args)
        print('rank{}: sampler initialized.'.format(self.global_rank))

    def _build_dataloader(self):
        dataset_config = self.config["dataset_config"]
        self.dataset_name = dataset_config["dataset_name"]
        dataset = getattr(dataset_module, dataset_config["dataset_name"], None)(dataset_config)

        self.dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            drop_last=False,
        )

    def _build_model(self):
        config_path = self.config["config_path"]
        checkpoint_path = self.config["checkpoint_path"]
        model_config = load_yaml(config_path)
        self.gaussian_diffusion = GaussianDiffusion(model_config["diffusion_config"], device=self.device)
        self.encoder = getattr(encoder_module, model_config["encoder_config"]["model"], None)(**model_config["encoder_config"])
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.encoder.load_state_dict(checkpoint['ema_encoder'])
        self.encoder = self.encoder.cuda()
        self.encoder.eval()

    def start(self):
        z_list = []

        with torch.inference_mode():
            for i, batch in enumerate(self.dataloader):
                print(i)
                x_0 = move_to_cuda(batch["x_0"])
                z = self.encoder(x_0)
                z_list.append(z.cpu())

            latent = torch.cat(z_list,dim=0)

            torch.save({"mean": latent.mean(0), "std":latent.std(0)}, "./"+ self.dataset_name.lower() + ".pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.config = {
        "config_path": "./trained-models/autoencoder/ffhq128/config.yml",
        "checkpoint_path": "./trained-models/autoencoder/ffhq128/checkpoint.pt",

        "dataset_config": {
            "dataset_name": "CELEBAHQ",
            "data_path": "./data/celebahq",
            "image_channel": 3,
            "image_size": 128,
            "augmentation": False,
        },

        "batch_size": 100,
        "num_workers": 2,
    }

    runner = Sampler(args)
    runner.start()
