"""
Module for calculating tournament variances.
"""

from .gini_index import Gini, gini
from .interquartile_range import IQR
from .interquartile_range import interquartile_range as iqr
from .metric import Metric
from .normalized_hhi import HICB, NormalizedHHI
from .normalized_hhi import herfindahl_hirschman_index as hhi
from .normalized_hhi import herfindahl_index_of_competitive_balance as hicb
from .normalized_hhi import normalized_herfindahl_hirschman_index as nhhi
from .variances import Variances

METRIC_MAP = {
    "variance": Variances,
    "nhhi": NormalizedHHI,
    "hicb": HICB,
    "gini": Gini,
    "iqr": IQR,
}

__all__ = [
    "Gini",
    "gini",
    "IQR",
    "iqr",
    "Metric",
    "HICB",
    "NormalizedHHI",
    "hhi",
    "hicb",
    "nhhi",
    "METRIC_MAP",
    "Variances",
]
