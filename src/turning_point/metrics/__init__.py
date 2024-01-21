"""
Module for calculating tournament variances.
"""

from .metric import Metric
from .normalized_hhi import HICB, NormalizedHHI
from .normalized_hhi import herfindahl_hirschman_index as hhi
from .normalized_hhi import herfindahl_index_of_competitive_balance as hicb
from .normalized_hhi import normalized_herfindahl_hirschman_index as nhhi
from .variances import Variances

METRIC_MAP = {"variance": Variances, "nhhi": NormalizedHHI, "hicb": HICB}

__all__ = [
    "Variances",
    "HICB",
    "NormalizedHHI",
    "hhi",
    "hicb",
    "nhhi",
    "Metric",
    "METRIC_MAP",
]
