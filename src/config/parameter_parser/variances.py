import random
from pathlib import Path

import numpy.random as nprandom
import pandas as pd

import turning_point.permutation_coefficient as pc
import turning_point.variance_stats as vs
from logs import log, turning_logger
from tournament_simulations.data_structures import Matches

from .. import types


def _get_variance_stats(matches: Matches, **kwargs) -> vs.ExpandingVarStats:
    """
    This function works both for real matches and permutation matches.

    For permutation matches it calculates stats for each permutation
    separately to reduce memory usage.
    """

    all_var_stats: list[pd.DataFrame] = []
    permutation_numbers = pc.get_permutation_numbers(matches.df)

    for str_number in permutation_numbers:
        turning_logger.info(f"Starting i-th permutation: {str_number}")

        filtered_matches = Matches(pc.get_ith_permutation(matches.df, str_number))
        var_stats = vs.ExpandingVarStats.from_matches(
            filtered_matches,
            id_to_probabilities=filtered_matches.probabilities_per_id,
            **kwargs,
        )

        all_var_stats.append(var_stats.df)

    return vs.ExpandingVarStats(pd.concat(all_var_stats).sort_index())


@log(turning_logger.info)
def _calculate_variance_stats(
    filenames: str | list[str],
    read_directory: Path,
    var_config: types.TurningPointConfig,
) -> dict[str, vs.ExpandingVarStats]:
    if not var_config["should_calculate_it"]:
        return {}

    random.seed(var_config["seed"])
    nprandom.seed(var_config["seed"])

    filenames = [filenames] if isinstance(filenames, str) else list(filenames)

    filename_to_var_stats = {}

    for filename in filenames:
        filepath = read_directory / f"{filename}.csv"
        if not filepath.exists():
            turning_logger.warning(f"No file: {filepath}")
            continue

        matches = Matches(pd.read_csv(filepath))

        kwargs = var_config["parameters"]
        var_stats = _get_variance_stats(matches, **kwargs)

        filename_to_var_stats[filename] = var_stats

    return filename_to_var_stats


def calculate_and_save_var_stats(
    config: types.RealConfig | types.PermutedConfig,
    read_directory: Path,
    save_directory: Path,
) -> None:
    filename_to_var_stats = _calculate_variance_stats(
        config["sports"],
        read_directory,
        config["turning_point"],
    )

    save_directory.mkdir(parents=True, exist_ok=True)
    for filename, var_stats in filename_to_var_stats.items():
        var_stats.df.to_csv(save_directory / f"{filename}.csv")
