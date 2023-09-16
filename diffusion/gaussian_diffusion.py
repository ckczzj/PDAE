import math
from functools import partial
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np

from diffusion.ddim import DDIM

class GaussianDiffusion:
    def __init__(self, config, device):
        super().__init__()
        self.device=device
        self.timesteps = config["timesteps"]
        betas_type = config["betas_type"]
        if betas_type == "linear":
            betas = np.linspace(0.0001, 0.02, self.timesteps)
        elif betas_type == "cosine":
            betas = []
            alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            max_beta = 0.999
            for i in range(self.timesteps):
                t1 = i / self.timesteps
                t2 = (i + 1) / self.timesteps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
            betas = np.array(betas)
        else:
            raise NotImplementedError

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.)

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        self.to_torch = to_torch

        self.alphas = to_torch(alphas)
        self.betas = to_torch(betas)
        self.alphas_cumprod = to_torch(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev)
        self.alphas_cumprod_next = to_torch(alphas_cumprod_next)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recip_alphas_cumprod_m1 = to_torch(np.sqrt(1. / alphas_cumprod - 1.))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = to_torch(posterior_variance)
        # clip the log because the posterior variance is 0 at the beginning of the diffusion chain
        posterior_log_variance_clipped = np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        self.posterior_log_variance_clipped = to_torch(posterior_log_variance_clipped)

        self.x_0_posterior_mean_x_0_coef = to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.x_0_posterior_mean_x_t_coef = to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.noise_posterior_mean_x_t_coef = to_torch(np.sqrt(1. / alphas))
        self.noise_posterior_mean_noise_coef = to_torch(betas/(np.sqrt(alphas)*np.sqrt(1. - alphas_cumprod)))

        self.shift_coef = to_torch( - np.sqrt(alphas) * (1. - alphas_cumprod_prev) / np.sqrt(1. - alphas_cumprod))
        # self.shift_coef_learn_sigma = to_torch( - np.sqrt(1. - alphas_cumprod) * np.sqrt(alphas) / betas)

        snr = alphas_cumprod / (1. - alphas_cumprod)
        gamma = 0.1
        self.weight = to_torch(snr ** gamma / (1. + snr))

    @staticmethod
    def extract_coef_at_t(schedule, t, x_shape):
        return torch.gather(schedule, -1, t).reshape([x_shape[0]] + [1] * (len(x_shape) - 1))

    @staticmethod
    def get_ddim_betas_and_timestep_map(ddim_style, original_alphas_cumprod):
        original_timesteps = original_alphas_cumprod.shape[0]
        dim_step = int(ddim_style[len("ddim"):])
        # data: x_{-1}  noisy latents: x_{0}, x_{1}, x_{2}, ..., x_{T-2}, x_{T-1}
        # encode: treat input x_{-1} as starting point x_{0}
        # sample: treat ending point x_{0} as output x_{-1}
        use_timesteps = set([int(s) for s in list(np.linspace(0, original_timesteps - 1, dim_step + 1))])
        timestep_map = []

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(original_alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)

        return np.array(new_betas), torch.tensor(timestep_map, dtype=torch.long)

    # x_start: batch_size x channel x height x width
    # t: batch_size
    def q_sample(self, x_0, t, noise):
        shape = x_0.shape
        return (
            self.extract_coef_at_t(self.sqrt_alphas_cumprod, t, shape) * x_0
            + self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape) * noise
        )

    def q_posterior_mean(self, x_0, x_t, t):
        shape = x_t.shape
        return self.extract_coef_at_t(self.x_0_posterior_mean_x_0_coef, t, shape) * x_0 \
               + self.extract_coef_at_t(self.x_0_posterior_mean_x_t_coef, t, shape) * x_t

    # x_t: batch_size x image_channel x image_size x image_size
    # t: batch_size
    def noise_p_sample(self, x_t, t, predicted_noise, learned_range=None):
        shape = x_t.shape
        predicted_mean = \
            self.extract_coef_at_t(self.noise_posterior_mean_x_t_coef, t, shape) * x_t - \
            self.extract_coef_at_t(self.noise_posterior_mean_noise_coef, t, shape) * predicted_noise
        if learned_range is not None:
            log_variance = self.learned_range_to_log_variance(learned_range, t)
        else:
            log_variance = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)

        noise = torch.randn(shape, device=self.device)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape([shape[0]] + [1] * (len(shape) - 1))
        return predicted_mean + nonzero_mask * (0.5 * log_variance).exp() * noise

    # x_t: batch_size x image_channel x image_size x image_size
    # t: batch_size
    def x_0_clip_p_sample(self, x_t, t, predicted_noise, learned_range=None, clip_x_0=True):
        shape = x_t.shape

        predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, predicted_noise)
        if clip_x_0:
            predicted_x_0.clamp_(-1,1)
        predicted_mean = self.q_posterior_mean(predicted_x_0, x_t, t)
        if learned_range is not None:
            log_variance = self.learned_range_to_log_variance(learned_range, t)
        else:
            log_variance = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)

        noise = torch.randn(shape, device=self.device)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape([shape[0]] + [1] * (len(shape) - 1))
        return predicted_mean + nonzero_mask * (0.5 * log_variance).exp() * noise

    def learned_range_to_log_variance(self, learned_range, t):
        shape = learned_range.shape
        min_log_variance = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)
        max_log_variance = self.extract_coef_at_t(torch.log(self.betas), t, shape)
        # The learned_range is [-1, 1] for [min_var, max_var].
        frac = (learned_range + 1) / 2
        return min_log_variance + frac * (max_log_variance - min_log_variance)

    def predicted_noise_to_predicted_x_0(self, x_t, t, predicted_noise):
        shape = x_t.shape
        return self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t \
               - self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise

    def predicted_noise_to_predicted_mean(self, x_t, t, predicted_noise):
        shape = x_t.shape
        return self.extract_coef_at_t(self.noise_posterior_mean_x_t_coef, t, shape) * x_t - \
               self.extract_coef_at_t(self.noise_posterior_mean_noise_coef, t, shape) * predicted_noise

    def p_loss(self, noise, predicted_noise, weight=None, loss_type="l2"):
        if loss_type == 'l1':
            return (noise - predicted_noise).abs().mean()
        elif loss_type == 'l2':
            if weight is not None:
                return torch.mean(weight * (noise - predicted_noise) ** 2)
            else:
                return torch.mean((noise - predicted_noise) ** 2)
        else:
            raise NotImplementedError()

    """
        test pretrained dpms
    """
    def test_pretrained_dpms(self, ddim_style, denoise_fn, x_T, condition=None):
        return self.ddim_sample(ddim_style, denoise_fn, x_T, condition)

    """
        ddim
    """
    def ddim_sample(self, ddim_style, denoise_fn, x_T, condition=None):
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.ddim_sample_loop(denoise_fn, x_T, condition)

    def ddim_encode(self, ddim_style, denoise_fn, x_0, condition=None):
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.ddim_encode_loop(denoise_fn, x_0, condition)

    """
        regular
    """
    def regular_train_one_batch(self, denoise_fn, x_0, condition=None):
        shape = x_0.shape
        batch_size = shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)
        predicted_noise = denoise_fn(x_t, t, condition)

        prediction_loss = self.p_loss(noise, predicted_noise)

        return {
            'prediction_loss': prediction_loss,
        }

    def regular_ddim_sample(self, ddim_style, denoise_fn, x_T, condition=None):
        return self.ddim_sample(ddim_style, denoise_fn, x_T, condition)

    def regular_ddpm_sample(self, denoise_fn, x_T, condition=None):
        shape = x_T.shape
        batch_size = shape[0]
        img = x_T
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            output = denoise_fn(img, t, condition)
            if output.shape[1] == 2 * shape[1]:
                predicted_noise, learned_range = torch.split(output, shape[1], dim=1)
            else:
                predicted_noise = output
                learned_range = None
            img = self.noise_p_sample(img, t, predicted_noise, learned_range)
        return img

    """
        representation learning
    """
    def representation_learning_train_one_batch(self, encoder, decoder, x_0):
        shape = x_0.shape
        batch_size = shape[0]

        z = encoder(x_0)

        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)

        predicted_noise, gradient = decoder(x_t, t, z)

        shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)

        # weight = None
        weight = self.extract_coef_at_t(self.weight, t ,shape)

        prediction_loss = self.p_loss(noise, predicted_noise + shift_coef * gradient, weight=weight)

        return {
            'prediction_loss': prediction_loss
        }

    def representation_learning_ddpm_sample(self, encoder, decoder, x_0, x_T, z=None):
        shape = x_0.shape
        batch_size = shape[0]

        if z is None:
            z = encoder(x_0)
        img = x_T

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            predicted_noise, gradient = decoder(img, t, z)
            shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)
            img = self.noise_p_sample(img, t, predicted_noise + shift_coef * gradient)
        return img


    def representation_learning_ddim_sample(self, ddim_style, encoder, decoder, x_0, x_T, z=None, stop_percent=0.0):
        if z is None:
            z = encoder(x_0)
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.shift_ddim_sample_loop(decoder, z, x_T, stop_percent=stop_percent)

    def representation_learning_ddim_encode(self, ddim_style, encoder, decoder, x_0, z=None):
        if z is None:
            z = encoder(x_0)
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.shift_ddim_encode_loop(decoder, z, x_0)

    def representation_learning_autoencoding(self, encoder_ddim_style, decoder_ddim_style, encoder, decoder, x_0):
        z = encoder(x_0)
        inferred_x_T = self.representation_learning_ddim_encode(encoder_ddim_style, encoder, decoder, x_0, z)
        return self.representation_learning_ddim_sample(decoder_ddim_style, None, decoder, None, inferred_x_T, z)

    def representation_learning_gap_measure(self, encoder, decoder, x_0):
        shape = x_0.shape
        batch_size = shape[0]
        z = encoder(x_0)

        predicted_posterior_mean_gap = []
        autoencoder_posterior_mean_gap = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x_t = self.q_sample(x_0, t, torch.rand_like(x_0))
            predicted_noise, gradient = decoder(x_t, t, z)

            predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, predicted_noise)
            predicted_posterior_mean = self.q_posterior_mean(predicted_x_0, x_t, t)

            shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)
            autoencoder_predicted_noise = predicted_noise + shift_coef * gradient
            autoencoder_predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, autoencoder_predicted_noise)
            autoencoder_predicted_posterior_mean = self.q_posterior_mean(autoencoder_predicted_x_0, x_t, t)

            true_posterior_mean = self.q_posterior_mean(x_0, x_t, t)

            predicted_posterior_mean_gap.append(torch.mean((true_posterior_mean - predicted_posterior_mean) ** 2, dim=[0, 1, 2, 3]).cpu().item())
            autoencoder_posterior_mean_gap.append(torch.mean((true_posterior_mean - autoencoder_predicted_posterior_mean) ** 2, dim=[0, 1, 2, 3]).cpu().item())

        return predicted_posterior_mean_gap, autoencoder_posterior_mean_gap

    def representation_learning_denoise_one_step(self, encoder, decoder, x_0, timestep_list):
        shape = x_0.shape

        t = torch.tensor(timestep_list, device=self.device, dtype=torch.long)
        x_t = self.q_sample(x_0, t, noise=torch.randn_like(x_0))
        z = encoder(x_0)
        predicted_noise, gradient = decoder(x_t, t, z)

        predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, predicted_noise)

        shift_coef = self.extract_coef_at_t(self.shift_coef, t, shape)
        autoencoder_predicted_noise = predicted_noise + shift_coef * gradient
        autoencoder_predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, autoencoder_predicted_noise)

        return predicted_x_0, autoencoder_predicted_x_0

    def representation_learning_ddim_trajectory_interpolation(self, ddim_style, decoder, z_1, z_2, x_T, alpha):
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.shift_ddim_trajectory_interpolation(decoder, z_1, z_2, x_T, alpha)

    """
        latent
    """
    @property
    def latent_diffusion_config(self):
        timesteps = 1000
        betas = np.array([0.008] * timesteps)
        # betas = np.linspace(0.0001, 0.02, timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        loss_type = "l1"

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        return {
            "timesteps": timesteps,
            "betas": betas,
            "alphas_cumprod": to_torch(alphas_cumprod),
            "sqrt_alphas_cumprod": to_torch(sqrt_alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": to_torch(sqrt_one_minus_alphas_cumprod),
            "loss_type": loss_type,
        }

    def normalize(self, z, mean, std):
        return (z - mean) / std


    def denormalize(self, z, mean, std):
        return z * std + mean


    def latent_diffusion_train_one_batch(self, latent_denoise_fn, encoder, x_0, latents_mean, latents_std):
        timesteps = self.latent_diffusion_config["timesteps"]

        sqrt_alphas_cumprod = self.latent_diffusion_config["sqrt_alphas_cumprod"]
        sqrt_one_minus_alphas_cumprod = self.latent_diffusion_config["sqrt_one_minus_alphas_cumprod"]

        z_0 = encoder(x_0)
        z_0 = z_0.detach()
        z_0 = self.normalize(z_0, latents_mean, latents_std)

        shape = z_0.shape
        batch_size = shape[0]

        t = torch.randint(0, timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(z_0)

        z_t = self.extract_coef_at_t(sqrt_alphas_cumprod, t, shape) * z_0 \
              + self.extract_coef_at_t(sqrt_one_minus_alphas_cumprod, t, shape) * noise

        predicted_noise = latent_denoise_fn(z_t, t)

        prediction_loss = self.p_loss(noise, predicted_noise, loss_type=self.latent_diffusion_config["loss_type"])

        return {
            'prediction_loss': prediction_loss,
        }

    def latent_diffusion_sample(self, latent_ddim_style, decoder_ddim_style, latent_denoise_fn, decoder, x_T, latents_mean, latents_std):
        alphas_cumprod = self.latent_diffusion_config["alphas_cumprod"]

        batch_size = x_T.shape[0]
        input_channel = latent_denoise_fn.input_channel
        z_T = torch.randn((batch_size, input_channel), device=self.device)

        z_T.clamp_(-1.0, 1.0) # may slightly improve sample quality

        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(latent_ddim_style, alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        z = ddim.latent_ddim_sample_loop(latent_denoise_fn, z_T)

        z = self.denormalize(z, latents_mean, latents_std)

        return self.representation_learning_ddim_sample(decoder_ddim_style, None, decoder, None, x_T, z, stop_percent=0.3)


    """
        manipulation
    """

    def manipulation_train_one_batch(self, classifier, encoder, x_0, label, latents_mean, latents_std):
        z = encoder(x_0)
        z = z.detach()
        z_norm = self.normalize(z, latents_mean, latents_std)

        prediction = classifier(z_norm)

        gt = torch.where(label > 0, torch.ones_like(label).float(), torch.zeros_like(label).float())
        loss = F.binary_cross_entropy_with_logits(prediction, gt)
        return {
            'bce_loss': loss,
        }

    def manipulation_sample(self, ddim_style, classifier_weight, encoder, decoder, x_0, inferred_x_T, latents_mean, latents_std, class_id, scale):
        z = encoder(x_0)
        z_norm = self.normalize(z, latents_mean, latents_std)

        import math
        z_norm_manipulated = z_norm + scale * math.sqrt(512) * F.normalize(classifier_weight[class_id][None,:], dim=1)
        z_manipulated = self.denormalize(z_norm_manipulated, latents_mean, latents_std)

        return self.representation_learning_ddim_sample(ddim_style, None, decoder, None, inferred_x_T, z_manipulated, stop_percent=0.0)
