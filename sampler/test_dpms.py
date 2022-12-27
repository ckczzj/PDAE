import sys
sys.path.append("../")

import torch

from model.diffusion import GaussianDiffusion
from model.unet import UNet
from utils import load_yaml, move_to_cuda, save_image

device = "cuda:0"
torch.cuda.set_device(device)

config = {
    "diffusion_config" : {
        "timesteps": 1000,
        "betas_type": "linear",
        "linear_beta_start": 0.0001,
        "linear_beta_end": 0.02
    },

    "config_path": "../pre-trained-dpms/ffhq128/config.yml",
    "checkpoint_path": "../pre-trained-dpms/ffhq128/checkpoint.pt",

    "image_channel": 3,
    "image_size": 128,
}

gaussian_diffusion = GaussianDiffusion(config["diffusion_config"], device=device)

config_path = config["config_path"]
checkpoint_path = config["checkpoint_path"]
model_config = load_yaml(config_path)
denoise_fn = UNet(**model_config["denoise_fn_config"])
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
denoise_fn.load_state_dict(checkpoint["ema_denoise_fn"])
denoise_fn = denoise_fn.cuda()
denoise_fn.eval()

image_channel = config["image_channel"]
image_size = config["image_size"]

with torch.inference_mode():
    x_T = move_to_cuda(torch.randn(9, image_channel, image_size, image_size))
    samples = gaussian_diffusion.test_pretrained_dpms(f'ddim100', denoise_fn, x_T)

    samples = samples.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
    samples = samples.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()

    save_image(samples, "./test_dpms_result.png")

# CUDA_VISIBLE_DEVICES=0 python3 test_dpms.py