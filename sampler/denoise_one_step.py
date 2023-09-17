import argparse
from PIL import Image

import torch

import dataset as dataset_module
from diffusion.gaussian_diffusion import GaussianDiffusion
import model.representation_learning.encoder as encoder_module
import model.representation_learning.decoder as decoder_module
from utils.utils import load_yaml, move_to_cuda

from sampler.base_sampler import BaseSampler

class Sampler(BaseSampler):
    def __init__(self, args):
        super().__init__(args)
        print('rank{}: sampler initialized.'.format(self.global_rank))

    def _build_dataloader(self):
        pass

    def _build_model(self):
        config_path = self.config["config_path"]
        checkpoint_path = self.config["checkpoint_path"]
        model_config = load_yaml(config_path)
        self.gaussian_diffusion = GaussianDiffusion(model_config["diffusion_config"], device=self.device)
        self.encoder = getattr(encoder_module, model_config["encoder_config"]["model"], None)(**model_config["encoder_config"])
        trained_ddpm_config = load_yaml(self.config["trained_ddpm_config_path"])
        self.decoder = getattr(decoder_module, model_config["decoder_config"]["model"], None)(latent_dim=model_config["decoder_config"]["latent_dim"], **trained_ddpm_config["denoise_fn_config"])
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.encoder.load_state_dict(checkpoint['ema_encoder'])
        self.decoder.load_state_dict(checkpoint['ema_decoder'])
        self.encoder = self.encoder.cuda()
        self.encoder.eval()
        self.decoder = self.decoder.cuda()
        self.decoder.eval()

    def start(self):
        dataset_config = self.config["dataset_config"]
        image_size = dataset_config["image_size"]
        dataset = getattr(dataset_module, dataset_config["dataset_name"], None)(dataset_config)

        image_index = self.config["image_index"]

        timestep_list = self.config["timestep_list"]

        with torch.inference_mode():
            data = dataset.__getitem__(image_index)
            gt = data["gt"]
            x_0 = move_to_cuda(data["x_0"])
            x_0 = x_0.repeat(len(timestep_list), 1, 1, 1)

            predicted_x_0, autoencoder_predicted_x_0 = self.gaussian_diffusion.representation_learning_denoise_one_step(self.encoder, self.decoder, x_0, timestep_list)

            predicted_x_0 = predicted_x_0.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
            predicted_x_0 = predicted_x_0.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            autoencoder_predicted_x_0 = autoencoder_predicted_x_0.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
            autoencoder_predicted_x_0 = autoencoder_predicted_x_0.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()

            merge = Image.new('RGB', ((len(timestep_list) + 1) * image_size, 2 * image_size), color=(255, 255, 255))

            merge.paste(Image.fromarray(gt), (0, int(0.5 * image_size)))

            for i,_ in enumerate(timestep_list):
                merge.paste(Image.fromarray(predicted_x_0[i]), ((i + 1) * image_size, 0))
                merge.paste(Image.fromarray(autoencoder_predicted_x_0[i]), ((i + 1) * image_size, image_size))

            merge.save("./denoise_one_step_result.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.config = {
        "config_path": "./trained-models/autoencoder/celebahq128/config.yml",
        "checkpoint_path": "./trained-models/autoencoder/celebahq128/checkpoint.pt",
        "trained_ddpm_config_path": "./pre-trained-dpms/celebahq128/config.yml",

        "dataset_config": {
            "dataset_name": "CELEBAHQ",
            "data_path": "./data/celebahq",
            "image_channel": 3,
            "image_size": 128,
            "augmentation": False,
        },

        "image_index": 23332,
        "timestep_list": [400, 500, 600, 700, 800]
    }

    runner = Sampler(args)
    runner.start()
