import torch
from torch import nn

from stable_worldmodel.wm.lewm.module import ConditionalBlock, Transformer

from .edm import NoiseEmbedding, precond_coeffs


class Denoiser(nn.Module):
    """EDM-preconditioned latent denoiser conditioned on context latents.

    Token layout (causal attention):
        [z_{t-H+1}, ..., z_t, z_noisy_{t+1}]
    AdaLN conditioning per position: action embedding + noise-level embedding.
    The noised token is conditioned on a_t (the action causing t -> t+1).
    """

    def __init__(
        self,
        *,
        history_size,
        input_dim,
        hidden_dim,
        depth,
        heads,
        mlp_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        sigma_data=1.0,
    ):
        super().__init__()
        self.history_size = history_size
        self.sigma_data = sigma_data
        self.pos_embedding = nn.Parameter(
            torch.randn(1, history_size + 1, input_dim)
        )
        self.noise_emb = NoiseEmbedding(input_dim)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, z_noisy_in, c_noise, ctx_emb, act_emb):
        """Raw network F_theta. Inputs are already c_in-scaled.

        z_noisy_in: (B, D) c_in-scaled noised next latent
        c_noise: (B, 1) noise conditioning
        ctx_emb: (B, T, D) clean context latents, T <= history_size
        act_emb: (B, T, D) action embeddings aligned with ctx_emb
        """
        T = ctx_emb.size(1)
        tokens = torch.cat([ctx_emb, z_noisy_in.unsqueeze(1)], dim=1)
        tokens = tokens + self.pos_embedding[:, : T + 1]
        # noised token gets the last context action (the one driving t -> t+1)
        cond = torch.cat([act_emb, act_emb[:, -1:]], dim=1)
        cond = cond + self.noise_emb(c_noise).unsqueeze(1)
        out = self.transformer(tokens, cond)
        return out[:, -1]

    def denoise(self, z_noisy, sigma, ctx_emb, act_emb):
        """Full preconditioned denoiser D(z; sigma) = c_skip*z + c_out*F(...).

        z_noisy: (B, D) at noise level sigma (B, 1)
        Returns the denoised estimate (B, D).
        """
        c_skip, c_out, c_in, c_noise = precond_coeffs(sigma, self.sigma_data)
        F_out = self.forward(c_in * z_noisy, c_noise, ctx_emb, act_emb)
        return c_skip * z_noisy + c_out * F_out
