"""EDM (Karras et al. 2022) utilities for the JEDI latent diffusion world model.

Reference: JEDI, https://arxiv.org/abs/2605.13013 (denoiser preconditioning,
log-normal noise sampling, few-step Euler sampler operating on latents).
"""

import math

import torch
from torch import nn


def latent_clamp(z, scale: float = 3.0):
    """Soft-clamp latents to (-scale, scale): C(z) = s * tanh(z / s)."""
    return torch.tanh(z / scale) * scale


def sample_sigma(n, p_mean: float, p_std: float, device=None, dtype=None):
    """Log-normal noise-level sampling: ln(sigma) ~ N(p_mean, p_std^2).

    Returns sigma of shape (n, 1) for broadcasting against (n, D) latents.
    """
    log_sigma = torch.randn(n, 1, device=device, dtype=dtype) * p_std + p_mean
    return log_sigma.exp()


def precond_coeffs(sigma, sigma_data: float):
    """EDM preconditioning coefficients.

    D(x; sigma) = c_skip * x + c_out * F(c_in * x, c_noise)

    Returns (c_skip, c_out, c_in, c_noise), each shaped like `sigma`.
    """
    var = sigma**2 + sigma_data**2
    c_skip = sigma_data**2 / var
    c_out = sigma * sigma_data / var.sqrt()
    c_in = 1.0 / var.sqrt()
    c_noise = sigma.log() / 4.0
    return c_skip, c_out, c_in, c_noise


def karras_sigmas(
    n_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device=None,
    dtype=None,
):
    """Descending Karras noise schedule with a final 0 appended.

    Returns a tensor of shape (n_steps + 1,).
    """
    ramp = torch.linspace(0, 1, n_steps, device=device, dtype=dtype)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device, dtype=dtype)])


def euler_sample(
    denoise_fn,
    shape,
    n_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device=None,
    dtype=None,
):
    """Few-step Euler sampler (DIAMOND / JEDI regime, e.g. n_steps=3).

    denoise_fn(x, sigma) -> denoised estimate D, with sigma of shape (B, 1).
    """
    sigmas = karras_sigmas(
        n_steps, sigma_min, sigma_max, rho, device=device, dtype=dtype
    )
    x = torch.randn(shape, device=device, dtype=dtype) * sigmas[0]
    for i in range(n_steps):
        sigma = sigmas[i].expand(shape[0], 1)
        d = (x - denoise_fn(x, sigma)) / sigmas[i]
        x = x + (sigmas[i + 1] - sigmas[i]) * d
    return x


class NoiseEmbedding(nn.Module):
    """Random Fourier features of c_noise followed by a small MLP."""

    def __init__(self, dim, fourier_dim=64, scale=16.0):
        super().__init__()
        self.register_buffer('freqs', torch.randn(fourier_dim // 2) * scale)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, c_noise):
        """
        c_noise: (B, 1) -> (B, dim)
        """
        x = c_noise.float().view(-1, 1) * self.freqs.view(1, -1) * 2 * math.pi
        emb = torch.cat([x.cos(), x.sin()], dim=-1)
        return self.mlp(emb.to(c_noise.dtype))


__all__ = [
    'NoiseEmbedding',
    'euler_sample',
    'karras_sigmas',
    'latent_clamp',
    'precond_coeffs',
    'sample_sigma',
]
