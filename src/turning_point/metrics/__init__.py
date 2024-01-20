"""
Module for calculating tournament variances.
"""

from .metric import Metric
from .variances import Variances

METRIC_MAP = {"variance": Variances}

__all__ = ["Variances", "Metric", "METRIC_MAP"]
