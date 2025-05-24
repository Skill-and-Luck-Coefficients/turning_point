from typing import Literal

import pandas as pd

from data_analysis.read_datasets import read_as_dicts
from turning_point.normal_coefficient import TurningPoint
from turning_point.permutation_coefficient import PermutationTurningPoint

TPType = TurningPoint | PermutationTurningPoint


def read_metrics_tp(
    sports: list[str],
    metrics: list[str],
    tp_kind: Literal["tp", "optimal_tp"],
    **read_as_dict_kwargs,
) -> dict[str, dict[str, TPType]]:
    """
    Read turning point for all desired metrics.

    ----
    Returns:
        KeySportMetric (dict[str, dict[str, TP]])
            ```
            {
                "{tp_kind}_{metric}": {
                    "{sport}": TurningPoint | PermutationTurningPoint
                }
            }
            ```
    """
    metrics_tp = {}

    for metric in metrics:
        read_data = read_as_dicts(sports, tp_kind, metric=metric, **read_as_dict_kwargs)

        key = f"{tp_kind}_{metric}"
        metrics_tp[key] = read_data[tp_kind]

    return metrics_tp


def get_tp_correlation(
    reference: str,
    to_compare: list[str],
    sports: list[str],
    tp_kind: Literal["tp", "optimal_tp"],
    tp_column: Literal["%turning point", "turning point"] = "%turning point",
    **read_as_dict_kwargs,
) -> pd.DataFrame:
    """
    Calculates the turning point correlation of `reference` to `to_compare`.
    """

    def _concat_metrics_one_sport(_sport: str):
        to_concat = {
            _metric_key: metrics_tp[f"{tp_kind}_{_metric_key}"][_sport].df
            for _metric_key in to_compare
        }
        _concated_metrics = pd.concat(to_concat, axis="columns")

        return _concated_metrics.swaplevel(0, 1, "columns").sort_index(axis="columns")

    all_metrics = [reference] + to_compare
    metrics_tp = read_metrics_tp(sports, all_metrics, tp_kind, **read_as_dict_kwargs)

    correlations = {
        sport: _concat_metrics_one_sport(sport)[tp_column].corr()[reference]
        for sport in sports
    }
    return pd.concat(correlations, axis="columns")
