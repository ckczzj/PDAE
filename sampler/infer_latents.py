import sys
sys.path.append("../")

import torch
from torch.utils.data import DataLoader

import dataset as dataset_module
from model.diffusion import GaussianDiffusion
import model.representation.encoder as encoder_module

from utils import load_yaml, move_to_cuda

device = "cuda:0"
torch.cuda.set_device(device)

config = {
    "config_path": "../trained-models/autoencoder/ffhq128/config.yml",
    "checkpoint_path": "../trained-models/autoencoder/ffhq128/checkpoint.pt",

    "dataset_name": "CELEBAHQ",
    "data_path": "../data/celebahq",
    "image_channel": 3,
    "image_size": 128,
}

config_path = config["config_path"]
checkpoint_path = config["checkpoint_path"]
model_config = load_yaml(config_path)
gaussian_diffusion = GaussianDiffusion(model_config["diffusion_config"], device=device)
encoder = getattr(encoder_module, model_config["encoder_config"]["model"], None)(**model_config["encoder_config"])
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
encoder.load_state_dict(checkpoint['ema_encoder'])
encoder = encoder.cuda()
encoder.eval()


dataset_name = config["dataset_name"]
data_path = config["data_path"]
image_size = config["image_size"]
image_channel = config["image_channel"]
dataset = getattr(dataset_module, dataset_name, None)({"data_path": data_path, "image_size": image_size, "image_channel": image_channel}, augmentation=False)

dataloader = DataLoader(dataset, shuffle=False, collate_fn=dataset.collate_fn, num_workers=0, batch_size=1000)

z_list = []

with torch.inference_mode():
    for i, batch in enumerate(dataloader):
        print(i)
        x_0 = move_to_cuda(batch["net_input"]["x_0"])
        z = encoder(x_0)
        z_list.append(z.cpu())

    latent = torch.cat(z_list,dim=0)

    torch.save({"mean": latent.mean(0), "std":latent.std(0)}, "./"+ dataset_name.lower() + ".pt")

# CUDA_VISIBLE_DEVICES=0 python3 infer_latents.py