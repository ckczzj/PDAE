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
    "augmentation": False,

    "image_index": 23332,
    "timestep_list": [400, 500, 600, 700, 800]
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
augmentation = config["augmentation"]
dataset = getattr(dataset_module, dataset_name, None)({"data_path": data_path, "image_size": image_size, "image_channel": image_channel, "augmentation": augmentation})

image_index = config["image_index"]

timestep_list = config["timestep_list"]

with torch.inference_mode():
    data = dataset.__getitem__(image_index)
    gt = data["gt"]
    x_0 = move_to_cuda(data["x_0"])
    x_0 = x_0.repeat(len(timestep_list), 1, 1, 1)

    predicted_x_0, autoencoder_predicted_x_0 = gaussian_diffusion.representation_learning_denoise_one_step(encoder, decoder, x_0, timestep_list)

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

# CUDA_VISIBLE_DEVICES=0 python3 denoise_one_step.py