"""
Language Pillar - Large Language Models & Text Classification
"""

from kitepy.core.adapters import register_pillar
from kitepy.pillars.language.data import SimpleCharTokenizer, load_language_data
from kitepy.pillars.language.models import LLM, TextClassifier, TinyGPT

register_pillar("language", data_loader=load_language_data, default_task="causal_lm")

__all__ = [
    "LLM",
    "TextClassifier",
    "TinyGPT",
    "SimpleCharTokenizer",
    "load_language_data",
]
