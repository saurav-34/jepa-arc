import torch

from stable_worldmodel.wm.lewm import LeWM

from .edm import karras_sigmas, latent_clamp


class JEDI(LeWM):
    """Joint Embedding Diffusion world model (https://arxiv.org/abs/2605.13013).

    LeWM with the deterministic predictor replaced by an EDM latent diffusion
    denoiser. `encode` additionally soft-clamps latents; `predict` runs a
    few-step Euler sampler, so `rollout`, `criterion` and `get_cost` are
    inherited unchanged (each plan sample gets its own stochastic rollout).
    """

    def __init__(
        self,
        encoder,
        denoiser,
        action_encoder,
        projector=None,
        latent_clamp: float = 3.0,
        sample_steps: int = 3,
        sigma_min: float = 2e-3,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        **kwargs,
    ):
        super().__init__(
            encoder=encoder,
            predictor=denoiser,
            action_encoder=action_encoder,
            projector=projector,
            **kwargs,
        )
        self.latent_clamp = latent_clamp
        self.sample_steps = sample_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    @property
    def denoiser(self):
        return self.predictor

    def encode(self, info):
        info = super().encode(info)
        info['emb'] = latent_clamp(info['emb'], self.latent_clamp)
        return info

    @torch.no_grad()
    def self_rollout(self, emb, act_emb, history: int = None):
        """Sample latents for frames history..T-1 autoregressively, feeding
        each sampled latent back as context (training-time conditioning
        switch: the denoiser sees its own outputs instead of encoder outputs).

        emb: (B, T, D) clean latents, act_emb: (B, T, A) aligned actions.
        Returns (B, T - history, D) sampled latents for frames H..T-1.
        """
        H = history or self.denoiser.history_size
        frames = list(emb[:, :H].unbind(1))
        for t in range(emb.size(1) - H):
            ctx = torch.stack(frames[t : t + H], dim=1)
            frames.append(self.predict(ctx, act_emb[:, t : t + H])[:, -1])
        return torch.stack(frames[H:], dim=1)

    def predict(self, emb, act_emb, n_steps: int = None):
        """Sample the next latent by few-step Euler denoising.

        emb: (B, T, D) clean context latents (T <= history_size)
        act_emb: (B, T, D) aligned action embeddings
        Returns (B, 1, D) so `rollout`'s `[:, -1]` indexing works unchanged.
        """
        n_steps = n_steps or self.sample_steps
        B, _, D = emb.shape
        sigmas = karras_sigmas(
            n_steps,
            self.sigma_min,
            self.sigma_max,
            self.rho,
            device=emb.device,
            dtype=emb.dtype,
        )
        x = torch.randn(B, D, device=emb.device, dtype=emb.dtype) * sigmas[0]
        for i in range(n_steps):
            sigma = sigmas[i].expand(B, 1)
            denoised = self.denoiser.denoise(x, sigma, emb, act_emb)
            d = (x - denoised) / sigmas[i]
            x = x + (sigmas[i + 1] - sigmas[i]) * d
        x = latent_clamp(x, self.latent_clamp)
        return x.unsqueeze(1)


__all__ = ['JEDI']
