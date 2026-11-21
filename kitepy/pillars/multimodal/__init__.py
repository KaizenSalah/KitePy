"""
Multimodal Pillar - Vision-Language Models (VLMs)
"""

from kitepy.core.adapters import register_pillar
from kitepy.pillars.multimodal.data import load_multimodal_data
from kitepy.pillars.multimodal.models import VLM, DualEncoderCLIP

register_pillar("multimodal", data_loader=load_multimodal_data, default_task="contrastive")

__all__ = ["VLM", "DualEncoderCLIP", "load_multimodal_data"]
