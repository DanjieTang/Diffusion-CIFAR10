import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(schedule: str, num_timesteps: int) -> torch.Tensor:
    """
    Build the variance (beta) schedule for the forward diffusion process.

    :param schedule: Either "linear" (DDPM) or "cosine" (improved DDPM).
    :param num_timesteps: Number of diffusion steps T.
    :return: Tensor of betas, shape (num_timesteps,).
    """
    if schedule == "linear":
        scale = 1000 / num_timesteps
        return torch.linspace(scale * 1e-4, scale * 0.02, num_timesteps, dtype=torch.float64)

    if schedule == "cosine":
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(max=0.999)

    raise ValueError(f"Unknown beta schedule: {schedule}")


def extract(values: torch.Tensor, timestep: torch.Tensor, broadcast_shape: torch.Size) -> torch.Tensor:
    """Gather per-sample scalars from a 1D buffer and reshape for broadcasting over images."""
    out = values.gather(0, timestep).float()
    return out.reshape(timestep.shape[0], *([1] * (len(broadcast_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """
    DDPM forward/reverse process with epsilon-prediction and classifier-free guidance.

    All schedule tensors are registered as buffers so they move with `.to(device)`.
    """

    def __init__(self, num_timesteps: int = 1000, beta_schedule: str = "linear"):
        super().__init__()
        self.num_timesteps = num_timesteps

        betas = make_beta_schedule(beta_schedule, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Posterior q(x_{t-1} | x_t, x_0) variance.
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        def register(name: str, tensor: torch.Tensor) -> None:
            self.register_buffer(name, tensor.float())

        register("betas", betas)
        register("alphas_cumprod", alphas_cumprod)
        register("alphas_cumprod_prev", alphas_cumprod_prev)
        register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        register("posterior_variance", posterior_variance)
        register("posterior_log_variance", torch.log(posterior_variance.clamp(min=1e-20)))
        register("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Diffuse x_start to timestep t: x_t = sqrt(a_bar) x_0 + sqrt(1 - a_bar) noise."""
        return (
            extract(self.sqrt_alphas_cumprod, timestep, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timestep, x_start.shape) * noise
        )

    def training_loss(self, model: nn.Module, x_start: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Sample a random timestep per image and return the epsilon-prediction MSE."""
        batch = x_start.shape[0]
        timestep = torch.randint(0, self.num_timesteps, (batch,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_noised = self.q_sample(x_start, timestep, noise)
        predicted_noise = model(x_noised, timestep, labels)
        return F.mse_loss(predicted_noise, noise)

    def predict_x_start_from_noise(self, x_t, timestep, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, timestep, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, timestep, x_t.shape) * noise
        )

    def _model_eps(self, model, x_t, timestep, labels, cfg_scale):
        """Run the model, applying classifier-free guidance when cfg_scale > 1."""
        if cfg_scale <= 1.0:
            return model(x_t, timestep, labels)

        null_labels = torch.full_like(labels, model.num_classes)
        combined_x = torch.cat([x_t, x_t], dim=0)
        combined_t = torch.cat([timestep, timestep], dim=0)
        combined_labels = torch.cat([labels, null_labels], dim=0)
        eps_cond, eps_uncond = model(combined_x, combined_t, combined_labels).chunk(2, dim=0)
        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    @torch.no_grad()
    def p_sample_loop(self, model, shape, labels, cfg_scale=1.0, device="cpu"):
        """Full ancestral DDPM sampling over all T timesteps."""
        image = torch.randn(shape, device=device)
        for step in reversed(range(self.num_timesteps)):
            timestep = torch.full((shape[0],), step, device=device, dtype=torch.long)
            eps = self._model_eps(model, image, timestep, labels, cfg_scale)
            x_start = self.predict_x_start_from_noise(image, timestep, eps).clamp(-1, 1)
            mean = (
                extract(self.posterior_mean_coef1, timestep, image.shape) * x_start
                + extract(self.posterior_mean_coef2, timestep, image.shape) * image
            )
            if step > 0:
                noise = torch.randn_like(image)
                log_var = extract(self.posterior_log_variance, timestep, image.shape)
                image = mean + (0.5 * log_var).exp() * noise
            else:
                image = mean
        return image

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, labels, cfg_scale=1.0, num_sampling_steps=50, eta=0.0, device="cpu"):
        """Deterministic-ish DDIM sampling on a strided subset of timesteps."""
        step_indices = torch.linspace(0, self.num_timesteps - 1, num_sampling_steps, dtype=torch.long)
        step_indices = list(reversed(step_indices.tolist()))

        image = torch.randn(shape, device=device)
        for i, step in enumerate(step_indices):
            timestep = torch.full((shape[0],), step, device=device, dtype=torch.long)
            eps = self._model_eps(model, image, timestep, labels, cfg_scale)
            x_start = self.predict_x_start_from_noise(image, timestep, eps).clamp(-1, 1)

            alpha_bar = extract(self.alphas_cumprod, timestep, image.shape)
            prev_step = step_indices[i + 1] if i + 1 < len(step_indices) else 0
            prev_timestep = torch.full((shape[0],), prev_step, device=device, dtype=torch.long)
            alpha_bar_prev = extract(self.alphas_cumprod, prev_timestep, image.shape)

            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
            )
            direction = torch.sqrt((1 - alpha_bar_prev - sigma ** 2).clamp(min=0)) * eps
            image = torch.sqrt(alpha_bar_prev) * x_start + direction
            if eta > 0 and i + 1 < len(step_indices):
                image = image + sigma * torch.randn_like(image)
        return image

    @torch.no_grad()
    def sample(self, model, num_images, labels, image_size, image_channel,
               sampler="ddpm", num_sampling_steps=50, cfg_scale=1.0, device="cpu"):
        """Convenience wrapper returning images in [-1, 1]."""
        shape = (num_images, image_channel, image_size, image_size)
        if sampler == "ddim":
            return self.ddim_sample_loop(model, shape, labels, cfg_scale, num_sampling_steps, device=device)
        return self.p_sample_loop(model, shape, labels, cfg_scale, device=device)
