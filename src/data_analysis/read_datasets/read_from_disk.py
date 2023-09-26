from pathlib import Path
from typing import Literal, Protocol, Sequence

import pandas as pd

import tournament_simulations.data_structures as ds
import turning_point.normal_coefficient as nc
import turning_point.permutation_coefficient as pc
import turning_point.variance_stats as vs
from config import path

Sport = Literal["basketball", "soccer", "handball", "volleyball"]

Key = Literal[
    "matches",
    "permuted_matches",
    "optimal_matches",
    "var_stats",
    "permuted_var_stats",
    "optimal_var_stats",
    "diff_points_var_stats",
    "tp",
    "permuted_tp",
    "optimal_tp",
    "diff_points_tp",
]


class ContainDF(Protocol):
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        ...


KEY_TO_CLASS_DIR: dict[Key, tuple[type[ContainDF], Path]] = {
    # Matches
    "matches": (ds.Matches, path.MATCHES_PATH),
    "permuted_matches": (ds.Matches, path.PERMUTED_MATCHES_PATH),
    "optimal_matches": (ds.Matches, path.OPTIMAL_MATCHES_PATH),
    # Variances
    "var_stats": (vs.ExpandingVarStats, path.VARIANCE_STATS_PATH),
    "permuted_var_stats": (vs.ExpandingVarStats, path.PERMUTED_VARIANCE_STATS_PATH),
    "optimal_var_stats": (vs.ExpandingVarStats, path.OPTIMAL_VARIANCE_STATS_PATH),
    "diff_points_var_stats": (vs.ExpandingVarStats, path.DIFF_POINTS_VAR_STATS_PATH),
    # Turning Point
    "tp": (nc.TurningPoint, path.TURNING_POINT_PATH),
    "permuted_tp": (pc.PermutationTurningPoint, path.PERMUTED_TURNING_POINT_PATH),
    "optimal_tp": (pc.PermutationTurningPoint, path.OPTIMAL_TURNING_POINT_PATH),
    "diff_points_tp": (nc.TurningPoint, path.DIFF_POINTS_TURN_POINT_PATH),
}


def read_as_dicts(
    sports: Sport | Sequence[Sport], dataset_keys: Key | Sequence[Key] | None = None
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
                Ranking Variances:
                    "var_stats": For real tournaments
                    "permuted_var_stats": For Permuted tournaments
                    "optimal_var_stats": Using an optimal schedule
                    "diff_points_var_stats": Using a different pointuation system
                Turning Point:
                    "tp": For real tournaments
                    "permuted_tp": For Permuted tournaments
                    "optimal_tp": Using an optimal schedule
                    "diff_points_tp": Using a different pointuation system

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

    return {
        key: {
            sport: class_(pd.read_csv(dir / f"{sport}.csv"))
            for sport in sports
            if Path(dir / f"{sport}.csv").exists()
        }
        for key, (class_, dir) in KEY_TO_CLASS_DIR.items()
        if key in set(dataset_keys)
    }
