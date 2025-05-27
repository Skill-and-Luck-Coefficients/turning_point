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

    Returns:

        MetricsTP (dict[str, dict[str, TP]]): Turning for all metrics and sports.
            ```
            {
                "{metric}": {
                    "{sport}": TurningPoint | PermutationTurningPoint
                }
            }
            ```
    """
    metrics_tp = {}

    for metric in metrics:
        read_data = read_as_dicts(sports, tp_kind, metric=metric, **read_as_dict_kwargs)
        metrics_tp[f"{metric}"] = read_data[tp_kind]

    return metrics_tp


def get_tp_correlation(
    reference: str,
    metrics_tp: dict[str, dict[str, PermutationTurningPoint]],
    filter_key: str | None = None,
    tp_column: Literal["%turning point", "turning point"] = "%turning point",
) -> pd.DataFrame:
    """
    Calculates correlation of `reference` to all metrics in `metrics_tp` segregated by sport.

    Parameters:

        reference (str):
            Which metric should be the reference.

        metrics_tp (dict[str, dict[str, TP]]): Turning for all metrics and sports.
            ```
            {
                "{metric}": {
                    "{sport}": TurningPoint | PermutationTurningPoint
                }
            }
            ```

        filter_key (str | None): How to filter the turning points for a given metric and sport.
            - (str) Filter only ids containing this key.
            - (None) Returns all available turning points.
    """

    def _get_sports():
        first_dict_entry = list(metrics_tp.values())[0]
        return first_dict_entry.keys()

    def _filter_tp(_tp: pd.DataFrame) -> pd.DataFrame:
        if filter_key is None:
            return _tp

        return _tp[_tp.index.get_level_values("id").str.contains(filter_key)]

    def _concat_metrics_one_sport(_sport: str):
        to_concat = {
            _key: _filter_tp(_sport_to_tp[_sport].df)
            for _key, _sport_to_tp in metrics_tp.items()
        }
        _concated_metrics = pd.concat(to_concat, axis="columns")
        return _concated_metrics.swaplevel(0, 1, "columns").sort_index(axis="columns")

    sports = _get_sports()

    correlations = {
        sport: _concat_metrics_one_sport(sport)[tp_column].corr()[reference]
        for sport in sports
    }
    return pd.concat(correlations, axis="columns")
