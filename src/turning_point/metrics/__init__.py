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
from .top_concentration_ratio import ConcentrationRatio
from .top_concentration_ratio import normalized_top_x_percent_concentration_ratio as ncr
from .top_concentration_ratio import top_x_percent_concentration_ratio as cr
from .variances import Variances

METRIC_MAP = {
    "variance": Variances,
    "nhhi": NormalizedHHI,
    "hicb": HICB,
    "gini": Gini,
    "iqr": IQR,
    "cr": ConcentrationRatio,
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
    "ConcentrationRatio",
    "cr",
    "ncr",
    "Variances",
    "METRIC_MAP",
]
