import random
from pathlib import Path

import numpy.random as nprandom
import pandas as pd

import turning_point.permutation_coefficient as pc
import turning_point.variance_stats as vs
from logs import log, turning_logger
from tournament_simulations.data_structures import Matches, PointsPerMatch

from .. import types


def _get_variance_stats(
    matches: Matches, quantile: float, **kwargs
) -> vs.ExpandingVarStats:
    """
    This function works both for real matches and permutation matches.

    For permutation matches it calculates stats for each permutation
    separately to reduce memory usage.
    """
    winner_to_points = {k: tuple(v) for k, v in kwargs["winner_to_points"].items()}
    point_pairs = sorted(set(winner_to_points.values()))

    all_var_stats: list[pd.DataFrame] = []
    permutation_ids = pc.get_permutation_identifiers(matches.df)

    for perm_id in permutation_ids:
        turning_logger.info(f"Starting i-th permutation: {perm_id}")

        filtered_matches = Matches(pc.get_data_with_identifier(matches.df, perm_id))

        # TODO: Remove this redundant calculation?
        filtered_ppm = PointsPerMatch.from_home_away_winner(
            home_away_winner=filtered_matches.home_away_winner(kwargs["winner_type"]),
            result_to_points=winner_to_points,
        )
        var_stats = vs.ExpandingVarStats.from_matches(
            filtered_matches,
            num_iteration_simulation=kwargs["num_iteration_simulation"],
            winner_type=kwargs["winner_type"],
            winner_to_points=winner_to_points,
            id_to_probabilities=filtered_ppm.probabilities_per_id(point_pairs),
            quantile=quantile,
        )

        all_var_stats.append(var_stats.df)

    return vs.ExpandingVarStats(pd.concat(all_var_stats).sort_index())


@log(turning_logger.info)
def _calculate_variance_stats(
    filenames: str | list[str],
    read_directory: Path,
    var_parameters: types.TurningPointParameters,
    quantile: float,
) -> dict[str, vs.ExpandingVarStats]:
    filenames = [filenames] if isinstance(filenames, str) else list(filenames)

    filename_to_var_stats = {}

    for filename in filenames:
        filepath = read_directory / f"{filename}.csv"
        if not filepath.exists():
            turning_logger.warning(f"No file: {filepath}")
            continue

        matches = Matches(pd.read_csv(filepath))
        var_stats = _get_variance_stats(matches, quantile, **var_parameters)

        filename_to_var_stats[filename] = var_stats

    return filename_to_var_stats


def _parse_quantiles_and_seeds(
    quantiles: float | list[float], seeds: int | list[int]
) -> tuple[list[float], list[int]]:
    if not isinstance(quantiles, list):
        quantiles = [quantiles]

    if not isinstance(seeds, list):
        seeds = [seeds]

    size_diff = len(quantiles) - len(seeds)
    if size_diff > 0:
        seeds = seeds + [seeds[0] for _ in range(size_diff)]

    return quantiles, seeds


def _get_quantile_path(original_path: Path, quantile: float) -> Path:
    if quantile == 0.95:
        return original_path
    return original_path / str(quantile)


def calculate_and_save_var_stats(
    config: types.RealConfig | types.PermutedConfig,
    read_directory: Path,
    save_directory: Path,
) -> None:
    var_config = config["turning_point"]

    if not var_config["should_calculate_it"]:
        return

    quantiles, seeds = _parse_quantiles_and_seeds(
        var_config["quantile"], var_config["seed"]
    )

    for seed, quantile in zip(seeds, quantiles):
        random.seed(seed)
        nprandom.seed(seed)

        filename_to_var_stats = _calculate_variance_stats(
            config["sports"],
            read_directory,
            config["turning_point"]["parameters"],
            quantile,
        )

        quantile_save_dir = _get_quantile_path(save_directory, quantile)
        quantile_save_dir.mkdir(parents=True, exist_ok=True)

        for filename, var_stats in filename_to_var_stats.items():
            var_stats.df.to_csv(quantile_save_dir / f"{filename}.csv")
