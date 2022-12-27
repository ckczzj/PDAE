import sys
sys.path.append("../")

from PIL import Image
import torch

import dataset as dataset_module
from model.diffusion import GaussianDiffusion
import model.representation.encoder as encoder_module
import model.representation.decoder as decoder_module
from utils import load_yaml, move_to_cuda

device = "cuda:0"
torch.cuda.set_device(device)

config = {
    "config_path": "../trained-models/autoencoder/celebahq128/config.yml",
    "checkpoint_path": "../trained-models/autoencoder/celebahq128/checkpoint.pt",

    "dataset_name": "CELEBAHQ",
    "data_path": "../data/celebahq",
    "image_channel": 3,
    "image_size": 128,

    "image_index": 29506,
}

config_path = config["config_path"]
checkpoint_path = config["checkpoint_path"]
model_config = load_yaml(config_path)
gaussian_diffusion = GaussianDiffusion(model_config["diffusion_config"], device=device)
encoder = getattr(encoder_module, model_config["encoder_config"]["model"], None)(**model_config["encoder_config"])
trained_ddpm_config = load_yaml(model_config["trained_ddpm_config"])
decoder = getattr(decoder_module, model_config["decoder_config"]["model"], None)(latent_dim=model_config["decoder_config"]["latent_dim"], **trained_ddpm_config["denoise_fn_config"])
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
encoder.load_state_dict(checkpoint['ema_encoder'])
decoder.load_state_dict(checkpoint['ema_decoder'])
encoder = encoder.cuda()
encoder.eval()
decoder = decoder.cuda()
decoder.eval()

dataset_name = config["dataset_name"]
data_path = config["data_path"]
image_size = config["image_size"]
image_channel = config["image_channel"]
dataset = getattr(dataset_module, dataset_name, None)({"data_path": data_path, "image_size": image_size, "image_channel": image_channel}, augmentation=False)

image_index = config["image_index"]

with torch.inference_mode():
    data = dataset.__getitem__(image_index)
    gt = data["gt"]
    x_0 = move_to_cuda(data["x_0"]).unsqueeze(0)

    reconstruction = gaussian_diffusion.representation_learning_autoencoding(f'ddim1000', f'ddim100', encoder, decoder, x_0)

    reconstruction = reconstruction.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
    reconstruction = reconstruction.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()

    x_0 = x_0.repeat(5, 1, 1, 1)
    ddpm_samples = gaussian_diffusion.representation_learning_ddpm_sample(encoder, decoder, x_0, torch.randn_like(x_0))
    ddpm_samples = ddpm_samples.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
    ddpm_samples = ddpm_samples.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()

    ddim_samples = gaussian_diffusion.representation_learning_ddim_sample(f"ddim100", encoder, decoder, x_0, torch.randn_like(x_0))
    ddim_samples = ddim_samples.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
    ddim_samples = ddim_samples.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()


    merge = Image.new('RGB', (12 * image_size + 8 , image_size), color = (255, 255, 255))

    merge.paste(Image.fromarray(gt), (0, 0))
    merge.paste(Image.fromarray(reconstruction[0]), (image_size, 0))


    for i in range(5):
        merge.paste(Image.fromarray(ddim_samples[i]), (4 + (i + 2) * image_size, 0))
        merge.paste(Image.fromarray(ddpm_samples[i]), (8 + (i + 7) * image_size, 0))

    merge.save("./autoencoding_example_result.png")

# CUDA_VISIBLE_DEVICES=0 python3 autoencoding_example.py