import torch
import numpy as np

from functools import partial
from tqdm import tqdm

class DDIM:
    def __init__(self, betas, timestep_map, device):
        super().__init__()
        self.device=device
        self.timestep_map = timestep_map.to(self.device)
        self.timesteps = betas.shape[0] - 1

        # length = timesteps + 1
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])  # 1. will never be used
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.)  # 0. will never be used

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        # self.alphas = to_torch(alphas)
        # self.betas = to_torch(betas)
        # self.alphas_cumprod = to_torch(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev)
        self.alphas_cumprod_next = to_torch(alphas_cumprod_next)


        # self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod))
        # self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recip_alphas_cumprod_m1 = to_torch(np.sqrt(1. / alphas_cumprod - 1.))

    @staticmethod
    def extract_coef_at_t(schedule, t, x_shape):
        return torch.gather(schedule, -1, t).reshape([x_shape[0]] + [1] * (len(x_shape) - 1))

    def t_transform(self, t):
        new_t = self.timestep_map[t]
        return new_t

    def ddim_sample(self, denoise_fn, x_t, t):
        shape = x_t.shape
        predicted_noise = denoise_fn(x_t, self.t_transform(t))
        predicted_x_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - \
                        self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise
        predicted_x_0 = predicted_x_0.clamp(-1, 1)

        new_predicted_noise = (self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - predicted_x_0) / \
              self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape)

        alpha_bar_prev = self.extract_coef_at_t(self.alphas_cumprod_prev, t, shape)

        return predicted_x_0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1. - alpha_bar_prev) * new_predicted_noise

    def ddim_sample_loop(self, denoise_fn, x_T):
        shape = x_T.shape
        batch_size = shape[0]
        img = x_T
        for i in tqdm(reversed(range(0 + 1, self.timesteps + 1)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.ddim_sample(denoise_fn, img, t)
        return img

    def ddim_encode(self, denoise_fn, x_t, t):
        shape = x_t.shape
        predicted_noise = denoise_fn(x_t, self.t_transform(t))
        predicted_x_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - \
                        self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise
        predicted_x_0 = predicted_x_0.clamp(-1, 1)

        new_predicted_noise = (self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - predicted_x_0) / \
                              self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape)

        alpha_bar_next = self.extract_coef_at_t(self.alphas_cumprod_next, t, shape)


        return predicted_x_0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * new_predicted_noise

    def ddim_encode_loop(self, denoise_fn, x_0):
        shape = x_0.shape
        batch_size = shape[0]
        x_t = x_0
        for i in tqdm(range(0, self.timesteps), desc='encoding loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x_t = self.ddim_encode(denoise_fn, x_t, t)
        return x_t


    def shift_ddim_sample(self, decoder, z, x_t, t, use_shift=True):
        shape = x_t.shape
        predicted_noise, gradient = decoder(x_t, self.t_transform(t), z)
        if use_shift:
            coef = self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape)
            predicted_noise = predicted_noise - coef * gradient

        predicted_x_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - \
                        self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise
        predicted_x_0 = predicted_x_0.clamp(-1, 1)

        new_predicted_noise = (self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - predicted_x_0) / \
              self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape)

        alpha_bar_prev = self.extract_coef_at_t(self.alphas_cumprod_prev, t, shape)

        return predicted_x_0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1. - alpha_bar_prev) * new_predicted_noise


    def shift_ddim_sample_loop(self, decoder, z, x_T, stop_percent=0.0):
        shape = x_T.shape
        batch_size = shape[0]
        img = x_T

        stop_step = int(stop_percent * self.timesteps)

        for i in tqdm(reversed(range(0 + 1, self.timesteps + 1)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.shift_ddim_sample(decoder, z, img, t, use_shift=True if (i - 1) >= stop_step else False)
        return img


    def shift_ddim_encode(self, decoder, z, x_t, t):
        shape = x_t.shape
        predicted_noise, gradient = decoder(x_t, self.t_transform(t), z)
        coef = self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape)
        predicted_noise = predicted_noise - coef * gradient

        predicted_x_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - \
                        self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise
        predicted_x_0 = predicted_x_0.clamp(-1, 1)

        new_predicted_noise = (self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - predicted_x_0) / \
                              self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape)

        alpha_bar_next = self.extract_coef_at_t(self.alphas_cumprod_next, t, shape)

        return predicted_x_0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * new_predicted_noise

    def shift_ddim_encode_loop(self, decoder, z, x_0):
        shape = x_0.shape
        batch_size = shape[0]
        x_t = x_0
        for i in tqdm(range(0, self.timesteps), desc='encoding loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x_t = self.shift_ddim_encode(decoder, z, x_t, t)
        return x_t

    def shift_ddim_trajectory_interpolation(self, decoder, z_1, z_2, x_T, alpha):
        shape = x_T.shape
        batch_size = shape[0]
        x_t = x_T

        for i in tqdm(reversed(range(0 + 1, self.timesteps + 1)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)

            predicted_noise, gradient_1 = decoder(x_t, self.t_transform(t), z_1)
            _, gradient_2 = decoder(x_t, self.t_transform(t), z_2)
            gradient = (1.0 - alpha) * gradient_1 + alpha * gradient_2
            coef = self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape)
            predicted_noise = predicted_noise - coef * gradient

            predicted_x_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - \
                            self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise
            predicted_x_0 = predicted_x_0.clamp(-1, 1)

            new_predicted_noise = (self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t - predicted_x_0) / \
                                  self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape)

            alpha_bar_prev = self.extract_coef_at_t(self.alphas_cumprod_prev, t, shape)

            x_t = predicted_x_0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1. - alpha_bar_prev) * new_predicted_noise

        return x_t



    def latent_ddim_sample(self, latent_denoise_fn, z_t, t):
        # shape = z_t.shape
        # predicted_noise = latent_denoise_fn(z_t, self.t_transform(t))
        # predicted_z_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * z_t - \
        #                 self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise
        # predicted_z_0.clamp_(-1.0, 1.0)
        # new_predicted_noise = (self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * z_t - predicted_z_0) / \
        #                       self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape)
        #
        # alpha_bar_prev = self.extract_coef_at_t(self.alphas_cumprod_prev, t, shape)
        #
        # return predicted_z_0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1. - alpha_bar_prev) * new_predicted_noise

        shape = z_t.shape
        predicted_noise = latent_denoise_fn(z_t, self.t_transform(t))
        predicted_z_0 = self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * z_t - \
                        self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise

        alpha_bar_prev = self.extract_coef_at_t(self.alphas_cumprod_prev, t, shape)

        return predicted_z_0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1. - alpha_bar_prev) * predicted_noise

    def latent_ddim_sample_loop(self, latent_denoise_fn, z_T):
        shape = z_T.shape
        batch_size = shape[0]
        z = z_T
        for i in tqdm(reversed(range(0 + 1, self.timesteps + 1)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            z = self.ddim_sample(latent_denoise_fn, z, t)
        return z
