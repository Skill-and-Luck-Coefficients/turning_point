from pathlib import Path
from typing import Literal, Protocol, Sequence

import pandas as pd

import tournament_simulations.data_structures as ds
import turning_point.metric_stats as ms
import turning_point.normal_coefficient as nc
import turning_point.permutation_coefficient as pc
from config import path

Sport = Literal["basketball", "soccer", "handball", "volleyball"] | str

Key = Literal[
    "matches",
    "permuted_matches",
    "optimal_matches",
    "bt_matches",
    "stats",
    "permuted_stats",
    "optimal_stats",
    "diff_points_stats",
    "bt_stats",
    "tp",
    "permuted_tp",
    "optimal_tp",
    "diff_points_tp",
    "bt_tp",
]


class ContainDF(Protocol):
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame): ...


KEY_TO_CLASS_DIR: dict[Key, tuple[type[ContainDF], Path]] = {
    # Matches
    "matches": (ds.Matches, path.MATCHES_PATH),
    "permuted_matches": (ds.Matches, path.PERMUTED_MATCHES_PATH),
    "optimal_matches": (ds.Matches, path.OPTIMAL_MATCHES_PATH),
    "bt_matches": (ds.Matches, path.BT_MATCHES_PATH),
    # Variances
    "stats": (ms.ExpandingMetricStats, path.STATS_PATH),
    "permuted_stats": (ms.ExpandingMetricStats, path.PERMUTED_STATS_PATH),
    "optimal_stats": (ms.ExpandingMetricStats, path.OPTIMAL_STATS_PATH),
    "diff_points_stats": (ms.ExpandingMetricStats, path.DIFF_POINTS_STATS_PATH),
    "bt_stats": (ms.ExpandingMetricStats, path.BT_STATS_PATH),
    # Turning Point
    "tp": (nc.TurningPoint, path.TURNING_POINT_PATH),
    "permuted_tp": (pc.PermutationTurningPoint, path.PERMUTED_TURNING_POINT_PATH),
    "optimal_tp": (pc.PermutationTurningPoint, path.OPTIMAL_TURNING_POINT_PATH),
    "diff_points_tp": (nc.TurningPoint, path.DIFF_POINTS_TURN_POINT_PATH),
    "bt_tp": (nc.TurningPoint, path.BT_TURNING_POINT_PATH),
}


def read_as_dicts(
    sports: Sport | Sequence[Sport],
    dataset_keys: Key | Sequence[Key] | None = None,
    quantile: float = 0.95,
    metric: str = "variance",
) -> dict[Key, dict[Sport, ContainDF]]:
    """
    Read desired dataset information from disk.

    ----
    Parameters:
        sports: Sport | Sequence[Sport]
            Desired sports.

            Sport: "basketball", "soccer", "handball", "volleyball"

        dataset_keys: Key | Sequence[Key] | None = None
            What data should be read.

            Key:
                Matches:
                    "matches": From real tournaments
                    "permuted_matches": From permuted tournaments
                    "optimal_matches": Following an optimal schedule
                    "bt_matches": From Bradley-Terry simulations
                Ranking Variances:
                    "stats": For real tournaments
                    "permuted_stats": For Permuted tournaments
                    "optimal_stats": Using an optimal schedule
                    "diff_points_stats": Using a different pointuation system
                    "bt_stats": From Bradley-Terry simulations
                Turning Point:
                    "tp": For real tournaments
                    "permuted_tp": For Permuted tournaments
                    "optimal_tp": Using an optimal schedule
                    "diff_points_tp": Using a different pointuation system
                    "bt_tp": From Bradley-Terry simulations

        quantile: float = 0.95
            Desired quantile value.

        metric: str = "variance"
            Desired metric.
    ----
    Returns:
        dict[Key, dict[Sport, <Desired Data>]]

    """
    if dataset_keys is None:
        dataset_keys = list(KEY_TO_CLASS_DIR.keys())

    if isinstance(sports, str):
        sports = [sports]
    if isinstance(dataset_keys, str):
        dataset_keys = [dataset_keys]

    key_to_sport_to_data = {}

    for key in dataset_keys:
        class_, dir_ = KEY_TO_CLASS_DIR[key]
        # Matches does not depend on quantile and metric
        if "matches" not in key:
            dir_ = dir_ / str(quantile) / metric

        sport_to_data = {}

        for sport in sports:
            path = dir_ / f"{sport}.csv"
            if path.exists():
                sport_to_data[sport] = class_(pd.read_csv(path))

        key_to_sport_to_data[key] = sport_to_data

    return key_to_sport_to_data
