"""
Audio Pillar - Speech and Audio Models
"""

from kitepy.core.adapters import register_pillar
from kitepy.pillars.audio.data import load_audio_data
from kitepy.pillars.audio.models import Audio, AudioCNN, AudioTransformer

register_pillar("audio", data_loader=load_audio_data, default_task="classification")

__all__ = ["Audio", "AudioCNN", "AudioTransformer", "load_audio_data"]
