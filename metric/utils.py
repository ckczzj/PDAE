import math
from PIL import Image

import torch
import torch.nn.functional as F
from torch.autograd import Variable

def numerical_rescale(x, is_0_1, to_0_1):
    if is_0_1 and to_0_1:
        return x.clamp(0., 1.).to(torch.float32)
    elif is_0_1 and not to_0_1:
        return ((x - 0.5) * 2.).clamp(-1., 1.).to(torch.float32)
    elif not is_0_1 and to_0_1:
        return ((x + 1.) / 2.).clamp(0., 1.).to(torch.float32)
    else:
        return x.clamp(-1., 1.).to(torch.float32)

def tensor_to_pillow(x, is_0_1):
    if not is_0_1:
        x = (x + 1.) / 2.
    x = x.mul(255).add(0.5).clamp(0, 255)
    x = x.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(x)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(1).mean(1).mean(1)

def calculate_ssim(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel)

def calculate_lpips(img1, img2, lpips_fn):
    return lpips_fn.forward(img1, img2).view(-1)

def calculate_mse(img1, img2):
    return (img1 - img2).pow(2).mean(dim=[1, 2, 3])

