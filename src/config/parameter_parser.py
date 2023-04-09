import json
import random
from pathlib import Path

import numpy.random as nprandom
import pandas as pd

import turning_point.normal_coefficient as nc
import turning_point.permutation_coefficient as pc
import turning_point.variance_stats as vs
from logs import log, turning_logger
from tournament_simulations.data_structures import Matches
from tournament_simulations.schedules.permutation import MatchesPermutations

from . import types


def read_json_configuration(path: Path) -> types.ConfigurationType:
    with open(path, "r") as config_file:
        configuration = json.load(config_file)

    return configuration


@log(turning_logger.info)
def _create_synthetic_matches(
    filenames: list[str],
    read_directory: Path,
    permuted_config: types.PermutedMatches,
) -> dict[str, Matches]:

    if not permuted_config["should_create_it"]:
        return {}

    random.seed(permuted_config["seed"])
    nprandom.seed(permuted_config["seed"])

    filename_to_matches = {}

    for filename in filenames:

        filepath = read_directory / f"{filename}.csv"
        if not filepath.exists():
            turning_logger.warning(f"No file: {filepath}")
            continue

        matches = Matches(pd.read_csv(filepath))
        permutations_creator = MatchesPermutations(matches)

        num_permutations = permuted_config["parameters"]["num_permutations"]
        permuted_matches = permutations_creator.create_n_permutations(num_permutations)

        filename_to_matches[filename] = permuted_matches

    return filename_to_matches


def create_and_save_permuted_matches(
    config: types.PermutedConfig,
    read_directory: Path,
    save_directory: Path,
) -> None:

    filename_to_matches = _create_synthetic_matches(
        config["sports"],
        read_directory,
        config["matches"],
    )

    save_directory.mkdir(parents=True, exist_ok=True)
    for filename, matches in filename_to_matches.items():
        matches.df.to_csv(save_directory / f"{filename}.csv")


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

        filtered_matches = Matches(pc.filter_ith_permutation(matches.df, str_number))
        var_stats = vs.ExpandingVarStats.from_matches(filtered_matches, **kwargs)

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


@log(turning_logger.info)
def _calculate_turning_point(
    filenames: str | list[str],
    read_directory: Path,
    tp_config: types.TurningPointConfig,
) -> dict[str, nc.TurningPoint]:

    if not tp_config["should_calculate_it"]:
        return {}

    filename_to_turning_point = {}

    for filename in filenames:

        filepath = read_directory / f"{filename}.csv"
        if not filepath.exists():
            turning_logger.warning(f"No file: {filepath}")
            continue

        var_stats = vs.ExpandingVarStats(pd.read_csv(filepath))

        # PermutationTurningPoints is the same when it comes to calculating it
        turning_point = nc.TurningPoint.from_expanding_var_stats(var_stats)
        filename_to_turning_point[filename] = turning_point

    return filename_to_turning_point


def calculate_and_save_turning_points(
    config: types.RealConfig | types.PermutedConfig,
    read_directory: Path,
    save_directory: Path,
) -> None:

    filename_to_turning_point = _calculate_turning_point(
        config["sports"],
        read_directory,
        config["turning_point"],
    )

    save_directory.mkdir(parents=True, exist_ok=True)
    for filename, turning_point in filename_to_turning_point.items():
        turning_point.df.to_csv(save_directory / f"{filename}.csv")
