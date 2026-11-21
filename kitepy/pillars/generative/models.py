"""
Generative Pillar - Convolutional VAE, Lightweight DDPM, and optional Diffusers UNet2D wrapper.
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from kitepy.core.adapters import get_adapter
from kitepy.core.base import BaseModel
from kitepy.core.config import GenerativeConfig


class ConvVAE(nn.Module):
    """Resolution-agnostic Convolutional Variational Autoencoder."""

    def __init__(self, in_channels: int = 3, img_size: int = 32, latent_dim: int = 32, base_channels: int = 32, kl_weight: float = 0.001):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        feat_dim = base_channels * 2 * 4 * 4
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, feat_dim)
        self.base_channels = base_channels

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        h = self.enc_conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc_dec(z)).view(z.size(0), self.base_channels * 2, 4, 4)
        out = self.dec_conv(h)
        if out.shape[-2:] != (self.img_size, self.img_size):
            out = F.interpolate(out, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return out

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int = 4, device: Optional[torch.device] = None) -> torch.Tensor:
        dev = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=dev)
        return self.decode(z)


class SimpleDDPM(nn.Module):
    """Lightweight Denoising Diffusion Probabilistic Model."""

    def __init__(self, in_channels: int = 3, img_size: int = 32, base_channels: int = 32, timesteps: int = 100):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.timesteps = timesteps

        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.time_emb = nn.Embedding(timesteps, base_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + base_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if t is None:
            t = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        t_emb = self.time_emb(t).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
        return self.net(torch.cat([x, t_emb], dim=1))

    def compute_loss(self, x: torch.Tensor):
        b = x.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x.device)
        noise = torch.randn_like(x)
        a_bar = self.alphas_cumprod[t].view(b, 1, 1, 1)
        x_t = torch.sqrt(a_bar) * x + torch.sqrt(1.0 - a_bar) * noise
        pred_noise = self.forward(x_t, t)
        return F.mse_loss(pred_noise, noise), pred_noise

    @torch.no_grad()
    def sample(self, n: int = 4, device: Optional[torch.device] = None) -> torch.Tensor:
        dev = device or self.betas.device
        x = torch.randn(n, self.in_channels, self.img_size, self.img_size, device=dev)
        for step in reversed(range(self.timesteps)):
            t = torch.full((n,), step, device=dev, dtype=torch.long)
            pred_noise = self.forward(x, t)
            alpha = self.alphas[step]
            alpha_bar = self.alphas_cumprod[step]
            beta = self.betas[step]
            coef = (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)
            x = (1.0 / torch.sqrt(alpha)) * (x - coef * pred_noise)
            if step > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        return x.clamp(0.0, 1.0)


class Generative(BaseModel):
    """
    Generative AI API supporting ConvVAE (`vae_small`, `vae_base`), `SimpleDDPM` (`ddpm_small`),
    and optional `diffusers` `UNet2DModel` when `diffusers` is installed.
    """

    modality = "generative"
    default_config_class = GenerativeConfig
    default_task = "generative"

    def __init__(
        self,
        model: Optional[Union[str, nn.Module]] = "vae_small",
        config: Optional[Union[Dict, GenerativeConfig, str]] = None,
        **kwargs,
    ):
        kwargs.setdefault("task", "generative")
        super().__init__(model, config, adapter=get_adapter("generative"), **kwargs)

    def _build_model(self):
        if self.custom_model is not None:
            self.model = self.custom_model
        elif self.model_name == "diffusers_unet":
            try:
                from diffusers import UNet2DModel
            except ImportError:
                raise ImportError("diffusers required for 'diffusers_unet'. Install with: pip install diffusers")
            self.model = UNet2DModel(
                sample_size=self.config.img_size,
                in_channels=self.config.in_channels,
                out_channels=self.config.in_channels,
            )
        elif self.config.arch == "ddpm":
            self.model = SimpleDDPM(
                in_channels=self.config.in_channels,
                img_size=self.config.img_size,
                base_channels=self.config.base_channels,
                timesteps=self.config.timesteps,
            )
        else:
            self.model = ConvVAE(
                in_channels=self.config.in_channels,
                img_size=self.config.img_size,
                latent_dim=self.config.latent_dim,
                base_channels=self.config.base_channels,
                kl_weight=self.config.kl_weight,
            )
        print(f"✓ Created Generative model: {self.model_name} ({sum(p.numel() for p in self.model.parameters()):,} params)")

    @torch.no_grad()
    def sample(self, n: int = 4) -> torch.Tensor:
        """Generate n new samples from the trained generative model."""
        if self.model is None:
            self._build_model()
        self.model.eval()
        device = self.device if self.device is not None else next(self.model.parameters()).device
        if hasattr(self.model, "sample"):
            return self.model.sample(n=n, device=device)
        raise NotImplementedError("Underlying model does not implement .sample(n)")


__all__ = ["Generative", "ConvVAE", "SimpleDDPM"]
