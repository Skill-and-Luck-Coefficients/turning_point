"""
Module for calculating tournament variances.
"""

from .gini_index import Gini, NormalizedGini
from .interquartile_range import IQR
from .metric import Metric
from .normalized_hhi import HICB, NaiveNormalizedHHI, NormalizedHHI
from .top_concentration_ratio import FastNormConcentrationRatio, NormConcentrationRatio
from .variances import Variances

METRIC_MAP: dict[str, Metric] = {
    "variance": Variances,
    "nhhi": NormalizedHHI,
    "naive_nhhi": NaiveNormalizedHHI,
    "hicb": HICB,
    "gini": Gini,
    "normalized_gini": NormalizedGini,
    "iqr": IQR,
    "ncr": NormConcentrationRatio,
    "fast_ncr": FastNormConcentrationRatio,
}

__all__ = [
    "NormalizedGini",
    "Gini",
    "IQR",
    "Metric",
    "HICB",
    "NaiveNormalizedHHI",
    "NormalizedHHI",
    "FastNormConcentrationRatio",
    "NormConcentrationRatio",
    "Variances",
    "METRIC_MAP",
]
