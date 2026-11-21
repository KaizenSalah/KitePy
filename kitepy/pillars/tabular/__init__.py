"""
Tabular Pillar - Tabular Classification & Regression Models
"""

from kitepy.core.adapters import register_pillar
from kitepy.pillars.tabular.data import load_tabular_data
from kitepy.pillars.tabular.models import Tabular, TabularMLP, TabularResNet

register_pillar("tabular", data_loader=load_tabular_data, default_task="classification")

__all__ = ["Tabular", "TabularResNet", "TabularMLP", "load_tabular_data"]
