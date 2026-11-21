"""
Generative Pillar - VAE & Diffusion Models
"""

from kitepy.core.adapters import register_pillar
from kitepy.pillars.generative.data import load_generative_data
from kitepy.pillars.generative.models import ConvVAE, Generative, SimpleDDPM

register_pillar("generative", data_loader=load_generative_data, default_task="generative")

__all__ = ["Generative", "ConvVAE", "SimpleDDPM", "load_generative_data"]
