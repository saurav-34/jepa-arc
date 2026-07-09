"""Tests for the JEDI world model (EDM utilities, denoiser, sampling)."""

import math

import pytest
import torch
from torch import nn

from stable_worldmodel.wm.jedi import JEDI
from stable_worldmodel.wm.jedi.edm import (
    karras_sigmas,
    latent_clamp,
    precond_coeffs,
    sample_sigma,
)
from stable_worldmodel.wm.jedi.module import Denoiser


D = 16
H = 3


def make_denoiser(sigma_data=1.0):
    return Denoiser(
        history_size=H,
        input_dim=D,
        hidden_dim=D,
        depth=1,
        heads=2,
        dim_head=8,
        mlp_dim=32,
        sigma_data=sigma_data,
    )


###############
## EDM utils ##
###############


def test_precond_coeffs_at_sigma_data():
    """At sigma == sigma_data the EDM coefficients take known closed forms."""
    sigma_data = 0.7
    sigma = torch.full((4, 1), sigma_data)
    c_skip, c_out, c_in, c_noise = precond_coeffs(sigma, sigma_data)
    assert torch.allclose(c_skip, torch.full_like(sigma, 0.5))
    assert torch.allclose(
        c_out, torch.full_like(sigma, sigma_data / math.sqrt(2))
    )
    assert torch.allclose(
        c_in, torch.full_like(sigma, 1 / (sigma_data * math.sqrt(2)))
    )
    assert torch.allclose(c_noise, sigma.log() / 4)


def test_precond_input_variance_is_one():
    """c_in normalizes the noised input to unit variance for any sigma."""
    sigma_data = 1.3
    sigma = torch.tensor([[0.01], [0.5], [10.0], [80.0]])
    _, _, c_in, _ = precond_coeffs(sigma, sigma_data)
    var = c_in**2 * (sigma**2 + sigma_data**2)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-6)


def test_sample_sigma_lognormal():
    torch.manual_seed(0)
    sigma = sample_sigma(20000, p_mean=-0.4, p_std=1.2)
    assert sigma.shape == (20000, 1)
    assert (sigma > 0).all()
    assert sigma.log().mean().item() == pytest.approx(-0.4, abs=0.05)
    assert sigma.log().std().item() == pytest.approx(1.2, abs=0.05)


def test_karras_sigmas_descending_with_final_zero():
    sigmas = karras_sigmas(5, sigma_min=2e-3, sigma_max=80.0)
    assert sigmas.shape == (6,)
    assert sigmas[0].item() == pytest.approx(80.0, rel=1e-5)
    assert sigmas[-2].item() == pytest.approx(2e-3, rel=1e-4)
    assert sigmas[-1].item() == 0.0
    assert (sigmas[:-1].diff() < 0).all()


def test_latent_clamp_bounds_and_identity_near_zero():
    z = torch.randn(100, D) * 100
    clamped = latent_clamp(z, 3.0)
    assert clamped.abs().max() <= 3.0
    small = torch.randn(100, D) * 0.01
    assert torch.allclose(latent_clamp(small, 3.0), small, atol=1e-5)


##############
## Denoiser ##
##############


def test_denoiser_forward_shape():
    den = make_denoiser()
    z_noisy = torch.randn(4, D)
    c_noise = torch.randn(4, 1)
    ctx = torch.randn(4, H, D)
    act = torch.randn(4, H, D)
    out = den(z_noisy, c_noise, ctx, act)
    assert out.shape == (4, D)


def test_denoiser_supports_shorter_context():
    den = make_denoiser()
    out = den(
        torch.randn(4, D),
        torch.randn(4, 1),
        torch.randn(4, 2, D),
        torch.randn(4, 2, D),
    )
    assert out.shape == (4, D)


def test_denoise_is_preconditioned_combination():
    den = make_denoiser(sigma_data=0.9)
    z_noisy = torch.randn(4, D)
    sigma = torch.rand(4, 1) + 0.1
    ctx, act = torch.randn(4, H, D), torch.randn(4, H, D)
    c_skip, c_out, c_in, c_noise = precond_coeffs(sigma, 0.9)
    torch.manual_seed(0)
    expected = c_skip * z_noisy + c_out * den(
        c_in * z_noisy, c_noise, ctx, act
    )
    torch.manual_seed(0)
    got = den.denoise(z_noisy, sigma, ctx, act)
    assert torch.allclose(got, expected, atol=1e-6)


##########
## JEDI ##
##########


class DummyEncoder(nn.Module):
    """Stands in for the HF ViT; JEDI tests exercise predict/rollout only."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def make_jedi(**kwargs):
    return JEDI(
        encoder=DummyEncoder(),
        denoiser=make_denoiser(),
        action_encoder=nn.Embedding(18, D),
        **kwargs,
    )


def test_jedi_predict_shape_and_clamp():
    model = make_jedi(latent_clamp=3.0, sample_steps=3)
    emb = torch.randn(4, H, D)
    act_emb = torch.randn(4, H, D)
    pred = model.predict(emb, act_emb)
    assert pred.shape == (4, 1, D)
    assert pred.abs().max() <= 3.0


def test_jedi_predict_is_stochastic():
    model = make_jedi()
    emb = torch.randn(4, H, D)
    act_emb = torch.randn(4, H, D)
    p1, p2 = model.predict(emb, act_emb), model.predict(emb, act_emb)
    assert not torch.allclose(p1, p2)


def test_jedi_rollout_shapes():
    model = make_jedi()
    B, S, T_hist, T_plan = 2, 5, 3, 7
    info = {
        'pixels': torch.zeros(B, S, T_hist, 3, 8, 8),
        'emb': torch.randn(B, S, T_hist, D),
    }
    actions = torch.randint(0, 18, (B, S, T_plan))
    info = model.rollout(info, actions)
    n_steps = T_plan - T_hist
    assert info['predicted_emb'].shape == (B, S, T_hist + n_steps + 1, D)


def test_jedi_self_rollout_shape_detached_and_clamped():
    """self_rollout samples T-H latents, without grad, inside the clamp."""
    model = make_jedi()
    T = 6
    emb = torch.randn(2, T, D, requires_grad=True)
    act_emb = torch.randn(2, T, D)
    out = model.self_rollout(emb, act_emb, history=H)
    assert out.shape == (2, T - H, D)
    assert not out.requires_grad
    assert out.abs().max() <= 3.0


def test_jedi_self_rollout_conditions_on_own_samples():
    """Later self_rollout frames must depend on earlier sampled frames."""
    z_history = []

    class RecordingDenoiser(nn.Module):
        sigma_data = 1.0

        def denoise(self, x, sigma, ctx, act):
            z_history.append(ctx)
            return torch.zeros_like(x)

    model = make_jedi(sample_steps=1)
    model.predictor = RecordingDenoiser()
    emb = torch.randn(2, 5, D)
    model.self_rollout(emb, torch.randn(2, 5, D), history=H)
    # window for frame 4 must contain the latent sampled for frame 3
    # (with an all-zero ideal denoiser the sampled frame is exactly 0)
    assert torch.allclose(z_history[-1][:, -1], torch.zeros(2, D), atol=1e-5)
    # and the real encoder frames are passed through unchanged
    assert torch.allclose(z_history[0], emb[:, :H])


def test_euler_sampler_recovers_point_mass():
    """With an ideal denoiser D(x) = z*, Euler sampling returns exactly z*."""
    model = make_jedi(sample_steps=8)
    z_star = torch.randn(4, D)

    class IdealDenoiser(nn.Module):
        sigma_data = 1.0

        def denoise(self, x, sigma, ctx, act):
            return z_star

    model.predictor = IdealDenoiser()
    pred = model.predict(torch.randn(4, H, D), torch.randn(4, H, D))
    assert torch.allclose(pred.squeeze(1), latent_clamp(z_star, 3.0), atol=1e-4)


def test_denoiser_overfits_single_transition():
    """The JEDI F-space loss decreases when overfitting one fixed transition."""
    torch.manual_seed(0)
    den = make_denoiser()
    opt = torch.optim.Adam(den.parameters(), lr=1e-3)
    ctx = torch.randn(8, H, D)
    act = torch.randn(8, H, D)
    z0 = torch.randn(8, D)

    losses = []
    for _ in range(200):
        sigma = sample_sigma(8, p_mean=-0.4, p_std=1.2)
        z_noisy = z0 + sigma * torch.randn_like(z0)
        c_skip, c_out, c_in, c_noise = precond_coeffs(sigma, den.sigma_data)
        f_out = den(c_in * z_noisy, c_noise, ctx, act)
        f_target = (z0 - c_skip * z_noisy) / c_out
        loss = (f_out - f_target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert sum(losses[-20:]) / 20 < 0.5 * sum(losses[:20]) / 20
