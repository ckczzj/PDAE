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

    def calculate_theta(self, a, b):
        return torch.arccos(torch.dot(a.view(-1), b.view(-1)) / (torch.norm(a) * torch.norm(b)))

    def slerp(self, a, b, alpha):
        theta = self.calculate_theta(a, b)
        sin_theta = torch.sin(theta)
        return a * torch.sin((1.0 - alpha) * theta) / sin_theta + b * torch.sin(alpha * theta) / sin_theta

    def lerp(self, a, b, alpha):
        return (1.0 - alpha) * a + alpha * b

    def start(self):
        dataset_config = self.config["dataset_config"]
        image_size = dataset_config["image_size"]
        dataset = getattr(dataset_module, dataset_config["dataset_name"], None)(dataset_config)

        image_index_1 = self.config["image_index_1"]
        image_index_2 = self.config["image_index_2"]

        with torch.inference_mode():
            data_1 = dataset.__getitem__(image_index_1)
            x_0_1 = move_to_cuda(data_1["x_0"]).unsqueeze(0)
            gt_1 = data_1["gt"]

            data_2 = dataset.__getitem__(image_index_2)
            x_0_2 = move_to_cuda(data_2["x_0"]).unsqueeze(0)
            gt_2 = data_2["gt"]

            z = self.encoder(torch.cat([x_0_1, x_0_2], dim=0))
            x_T = self.gaussian_diffusion.representation_learning_ddim_encode(
                f'ddim100',
                self.encoder,
                self.decoder,
                torch.cat([x_0_1, x_0_2], dim=0),
                z
            )

            x_T_1 = x_T[0:1]
            x_T_2 = x_T[1:2]
            z_1 = z[0:1]
            z_2 = z[1:2]

            merge = Image.new('RGB', (13 * image_size, 2 * image_size), color=(255, 255, 255))

            merge.paste(Image.fromarray(gt_1), (0, int(0.5 * image_size)))
            merge.paste(Image.fromarray(gt_2), (12 * image_size, int(0.5 * image_size)))

            for i, alpha in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
                x_T = self.slerp(x_T_1, x_T_2, alpha)
                z = self.lerp(z_1, z_2, alpha)

                image = self.gaussian_diffusion.representation_learning_ddim_sample(f'ddim100', None, self.decoder, None, x_T, z)

                image = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
                image = image.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                merge.paste(Image.fromarray(image[0]), ((i + 1) * image_size, 0))

            for i, alpha in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
                x_T = self.slerp(x_T_1, x_T_2, alpha)

                image = self.gaussian_diffusion.representation_learning_ddim_trajectory_interpolation(f'ddim100', self.decoder, z_1, z_2, x_T, alpha)

                image = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
                image = image.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                merge.paste(Image.fromarray(image[0]), ((i + 1) * image_size, image_size))

            merge.save("./interpolation_result.png")

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

        "image_index_1": 17570,
        "image_index_2": 23404,
    }

    runner = Sampler(args)
    runner.start()
