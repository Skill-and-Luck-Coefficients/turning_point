from dataclasses import dataclass
from itertools import product

import pandas as pd

from turning_point.normal_coefficient import TurningPoint

from .permutation_turning_point import PermutationTurningPoint
from .utils import add_second_level_to_column_names


def _parse_permutation(
    permutation_tp: PermutationTurningPoint, percentiles: list[float]
) -> tuple[pd.DataFrame, list[str]]:
    """
    ---
    Returns:
        tuple[
            pd.Dataframe,   # permutation stats
            list[str],      # column names
        ]
    """
    permutation_stats, stats_cols = pd.DataFrame(), []

    if permutation_tp is not None:
        permutation_stats = permutation_tp.statistical_measures(percentiles)
        stats_cols = permutation_stats["turning point"].columns.to_list()

    return permutation_stats, stats_cols


def _parse_optimal(
    optimal_tp: PermutationTurningPoint,
) -> tuple[pd.DataFrame, list[str]]:
    """
    ---
    Returns:
        tuple[
            pd.Dataframe,   # optimal turning point columns
            list[str],      # column names
        ]
    """
    optimal, optimal_cols = pd.DataFrame(), []

    if optimal_tp is not None:
        optimal = optimal_tp.unstack_permutation_id()
        optimal_cols = optimal["turning point"].columns.to_list()

    return optimal, optimal_cols


def get_turning_point_comparison(
    normal: TurningPoint,
    permutation_tp: PermutationTurningPoint | None = None,
    optimal_tp: PermutationTurningPoint | None = None,
    percentiles: list[str] | None = None,
) -> pd.DataFrame:
    """
    Join:
        - normal turning point values
        - statistical measures from permutation turning point (if not None)
        - optimal turning point values (if not None)

    Parameters:
        percentiles: list[float] | None = None
            Percentile values to calculate. They should be between 0 and 100.

            Only necessary if `permutation_tp` is not None.

    ----
    Returns:
        pd.DataFrame[
            index=[
                id"    -> "{current_name}@/{sport}/{country}/{name-year}"
            ],\n
            columns=[  # multi level columns
                "turning point"  -> turning point (int),\n
                    "normal"        -> turning point from original schedule,\n
                    "mean"          -> mean turning point,\n                   # from `permutation_tp`
                    "std"           -> standard deviation,\n                   # from `permutation_tp`
                    "f{p}%"         -> percentiles: p in 'percentiles'         # from `permutation_tp`
                    "{optimal_cols} -> all columns from optimal schedule,\n    # from `optimal_tp`
                "%turning point" -> normalized turning point (percentage),\n
                    "normal"        -> turning point from original schedule,\n
                    "mean"          -> mean turning point,\n                   # from `permutation_tp`
                    "std"           -> standard deviation,\n                   # from `permutation_tp`
                    "f{p}%"         -> percentiles: p in 'percentiles'         # from `permutation_tp`
                    "{optimal_cols} -> all columns from optimal schedule,\n    # from `optimal_tp`
        ]
    """
    extended_normal = add_second_level_to_column_names(normal.df, "normal")

    permutation_stats, stats_cols = _parse_permutation(permutation_tp, percentiles)
    optimal, optimal_cols = _parse_optimal(optimal_tp)

    data_type = {"id": "category"}
    joined = (
        pd.concat([extended_normal, permutation_stats, optimal], axis=1)
        .reset_index("id")
        .astype(data_type)
        .set_index("id")
    )

    col_level_one_order = ["turning point", "%turning point"]
    col_level_two_order = ["normal"] + stats_cols + optimal_cols
    col_order = list(product(col_level_one_order, col_level_two_order))
    return joined.sort_index()[col_order]
