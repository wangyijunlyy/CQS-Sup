"""Dreamer-style latent-dynamics world model for quantum expectation values.

The circuit layers play the role of time: per-layer parameters x_l are embedded
by an Encoder, a recurrent RSSM cell rolls the latent state forward across
layers, and a Predictor maps the final latent state to the expectation values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Per-layer parameter block -> embedding."""

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LatentRSSM(nn.Module):
    """Recurrent latent dynamics over circuit layers.

    h_l = GRU(h_{l-1}, Encoder(x_l)); optionally a stochastic latent is
    sampled from the deterministic state (Dreamer-style extension, off by
    default to keep v1 robust).
    """

    def __init__(
        self,
        emb_dim: int,
        hidden: int = 128,
        stochastic: bool = False,
        beta: float = 1e-3,
    ):
        super().__init__()
        self.hidden = hidden
        self.stochastic = stochastic
        self.beta = beta
        self.gru = nn.GRUCell(emb_dim, hidden)
        if stochastic:
            self.stoch_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2 * hidden)
            )
            self.latent_embed = nn.Linear(hidden, hidden)

    def forward(self, emb: torch.Tensor, return_all: bool = False):
        """emb: [B, L, emb_dim] -> (h_L [B, hidden] or [B, L, hidden], kl [B] or None)."""
        B, L, _ = emb.shape
        h = torch.zeros(B, self.hidden, device=emb.device)
        kl = torch.zeros(B, device=emb.device) if self.stochastic else None
        hs = [] if return_all else None
        for layer in range(L):
            h = self.gru(emb[:, layer], h)
            if self.stochastic:
                mu_logvar = self.stoch_head(h)
                mu, logvar = mu_logvar.chunk(2, dim=-1)
                logvar = torch.clamp(logvar, -10.0, 2.0)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                kl = kl + 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar).sum(dim=-1)
                h = h + self.latent_embed(z)
            if return_all:
                hs.append(h)
        if return_all:
            return torch.stack(hs, dim=1), kl
        return h, kl


class Predictor(nn.Module):
    """Final latent state -> expectation values."""

    def __init__(self, hidden: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


class WorldModel(nn.Module):
    """Encoder + RSSM + Predictor."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        encoder_hidden: int = 128,
        rssm_hidden: int = 128,
        predictor_hidden: int = 128,
        stochastic: bool = False,
        beta: float = 1e-3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = Encoder(in_dim, encoder_hidden, dropout)
        self.rssm = LatentRSSM(encoder_hidden, rssm_hidden, stochastic, beta)
        self.predictor = Predictor(predictor_hidden or rssm_hidden, out_dim)

    def forward(self, x: torch.Tensor):
        """x: [B, L, in_dim] -> (y_hat [B, out_dim], kl or None)."""
        emb = self.encoder(x)
        h, kl = self.rssm(emb)
        return self.predictor(h), kl

    def loss(self, x: torch.Tensor, y: torch.Tensor):
        y_hat, kl = self.forward(x)
        mse = F.mse_loss(y_hat, y)
        if kl is not None:
            total = mse + self.rssm.beta * kl.mean()
            return total, float(mse.item()), float(kl.mean().item())
        return mse, float(mse.item()), None


class ObsWorldModel(nn.Module):
    """Observation-driven multi-step world model (classical shadows as input).

    At each time step k (driving period), the observation encoder embeds the
    classical-shadow record o_k, the RSSM rolls the latent state forward, and the
    predictor outputs the magnetization <Z(k)> from the latent at step k.
    All K steps are supervised (multi-step loss).
    """

    def __init__(
        self,
        obs_dim: int,
        out_dim: int,
        encoder_hidden: int = 128,
        rssm_hidden: int = 128,
        predictor_hidden: int = 128,
        stochastic: bool = False,
        beta: float = 1e-3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = Encoder(obs_dim, encoder_hidden, dropout)
        self.rssm = LatentRSSM(encoder_hidden, rssm_hidden, stochastic, beta)
        self.predictor = Predictor(predictor_hidden or rssm_hidden, out_dim)

    def forward(self, obs: torch.Tensor):
        """obs: [B, K, obs_dim] -> (y_hat [B, K, out_dim], kl or None)."""
        emb = self.encoder(obs)
        hs, kl = self.rssm(emb, return_all=True)
        return self.predictor(hs), kl

    def loss(self, obs: torch.Tensor, y: torch.Tensor):
        y_hat, kl = self.forward(obs)
        mse = F.mse_loss(y_hat, y)
        if kl is not None:
            total = mse + self.rssm.beta * kl.mean()
            return total, float(mse.item()), float(kl.mean().item())
        return mse, float(mse.item()), None
