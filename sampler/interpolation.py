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

    "image_index_1": 17570,
    "image_index_2": 23404,
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

def calculate_theta(a,b):
    return torch.arccos(torch.dot(a.view(-1),b.view(-1))/(torch.norm(a)*torch.norm(b)))

def slerp(a,b,alpha):
    theta = calculate_theta(a,b)
    sin_theta = torch.sin(theta)
    return a * torch.sin((1.0 - alpha)*theta) / sin_theta + b * torch.sin(alpha * theta) / sin_theta

def lerp(a,b,alpha):
    return (1.0 - alpha) * a + alpha * b


image_index_1 = config["image_index_1"]
image_index_2 = config["image_index_2"]

with torch.inference_mode():
    data_1 = dataset.__getitem__(image_index_1)
    x_0_1 = move_to_cuda(data_1["x_0"]).unsqueeze(0)
    gt_1 = data_1["gt"]

    data_2 = dataset.__getitem__(image_index_2)
    x_0_2 = move_to_cuda(data_2["x_0"]).unsqueeze(0)
    gt_2 = data_2["gt"]

    z = encoder(torch.cat([x_0_1, x_0_2], dim=0))
    x_T = gaussian_diffusion.representation_learning_ddim_encode(
        f'ddim1000',
        encoder,
        decoder,
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
        x_T = slerp(x_T_1, x_T_2, alpha)
        z = lerp(z_1, z_2, alpha)

        image = gaussian_diffusion.representation_learning_ddim_sample(f'ddim100', None, decoder, None, x_T, z)

        image = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
        image = image.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        merge.paste(Image.fromarray(image[0]), ((i + 1) * image_size, 0))

    for i, alpha in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        x_T = slerp(x_T_1, x_T_2, alpha)

        image = gaussian_diffusion.representation_learning_ddim_trajectory_interpolation(f'ddim100', decoder, z_1, z_2, x_T, alpha)

        image = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
        image = image.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        merge.paste(Image.fromarray(image[0]), ((i + 1) * image_size, image_size))

    merge.save("./interpolation_result.png")

# CUDA_VISIBLE_DEVICES=0 python3 interpolation.py