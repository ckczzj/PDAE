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

        checkpoint = torch.load(self.config["classifier_checkpoint_path"], map_location=torch.device('cpu'))
        self.classifier = torch.nn.Linear(512, 40)
        self.classifier.load_state_dict(checkpoint['ema_classifier'])
        self.classifier = self.classifier.cuda()

    def start(self):
        id_to_label = [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
            'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
            'Wearing_Necklace', 'Wearing_Necktie', 'Young'
        ]
        label_to_id = {v: k for k, v in enumerate(id_to_label)}

        # load latents stat
        inferred_latents = torch.load(self.config["inferred_latents_path"], map_location=torch.device('cpu'))
        latents_mean = inferred_latents["mean"].cuda()
        latents_std = inferred_latents["std"].cuda()

        dataset_config = self.config["dataset_config"]
        image_size = dataset_config["image_size"]
        dataset = getattr(dataset_module, dataset_config["dataset_name"], None)(dataset_config)

        image_index = self.config["image_index"]

        class_id = label_to_id[self.config["attribute"]]
        scale_list = self.config["scale_list"]

        with torch.inference_mode():
            data = dataset.__getitem__(image_index)
            gt = data["gt"]
            x_0 = move_to_cuda(data["x_0"])
            x_0 = x_0.unsqueeze(0)

            inferred_x_T = self.gaussian_diffusion.representation_learning_ddim_encode(f'ddim500', self.encoder, self.decoder, x_0)
            result_list = []
            for scale in scale_list:
                result = self.gaussian_diffusion.manipulation_sample(
                    classifier_weight=self.classifier.weight,
                    encoder=self.encoder,
                    decoder=self.decoder,
                    x_0=x_0,
                    inferred_x_T=inferred_x_T,
                    latents_mean=latents_mean,
                    latents_std=latents_std,
                    class_id=class_id,
                    scale=scale,
                    ddim_style=f'ddim200',
                )
                result = result.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
                result = result.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                result_list.append(result)

            merge = Image.new('RGB', ((len(scale_list) + 1) * image_size, image_size), color=(255, 255, 255))
            for i in range(len(scale_list) // 2):
                merge.paste(Image.fromarray(result_list[i][0]), (i * image_size, 0))
            merge.paste(Image.fromarray(gt), (len(scale_list) // 2 * image_size, 0))
            for i in range(len(scale_list) // 2, len(scale_list)):
                merge.paste(Image.fromarray(result_list[i][0]), ((i + 1) * image_size, 0))
            merge.save("./manipulation_result.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.config = {
        "config_path": "./trained-models/autoencoder/celebahq128/config.yml",
        "checkpoint_path": "./trained-models/autoencoder/celebahq128/checkpoint.pt",
        "trained_ddpm_config_path": "./pre-trained-dpms/celebahq128/config.yml",

        "inferred_latents_path": "./trained-models/latents/celebahq.pt",
        "classifier_checkpoint_path": "./trained-models/classifier/checkpoint.pt",

        "dataset_config": {
            "dataset_name": "CELEBAHQ",
            "data_path": "./data/celebahq",
            "image_channel": 3,
            "image_size": 128,
            "augmentation": False,
        },

        "image_index": 14340,
        "attribute": "No_Beard",
        "scale_list": [-0.3, -0.1, 0.1, 0.3],
    }

    runner = Sampler(args)
    runner.start()
