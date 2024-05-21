import random
from pathlib import Path

import numpy.random as nprandom
import pandas as pd

import turning_point.metric_stats as ms
import turning_point.permutation_coefficient as pc
from logs import log, turning_logger
from tournament_simulations.data_structures import Matches, PointsPerMatch
from turning_point.metrics import METRIC_MAP

from .. import types
from . import utils


def _get_metric_stats(
    matches: Matches, quantile: float, metric: str, **kwargs
) -> ms.ExpandingMetricStats:
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
        var_stats = ms.ExpandingMetricStats.from_matches(
            filtered_matches,
            num_iteration_simulation=kwargs["num_iteration_simulation"],
            winner_type=kwargs["winner_type"],
            winner_to_points=winner_to_points,
            id_to_probabilities=filtered_ppm.probabilities_per_id(point_pairs),
            quantile=quantile,
            metric_type=METRIC_MAP[metric],
        )

        all_var_stats.append(var_stats.df)

    return ms.ExpandingMetricStats(pd.concat(all_var_stats).sort_index())


@log(turning_logger.info)
def _calculate_metric_stats(
    filepath: Path,
    var_parameters: types.TurningPointParameters,
    quantile: float,
    metric: str,
) -> ms.ExpandingMetricStats:
    matches = Matches(pd.read_csv(filepath))
    return _get_metric_stats(matches, quantile, metric, **var_parameters)


def _extend_seeds_as_quantiles(
    quantiles: list[float],
    seeds: list[int],
) -> list[int]:
    """
    Make `seeds` have the same length as `quantiles`
    """
    size_diff = len(quantiles) - len(seeds)

    if size_diff > 0:
        seeds = seeds + [seeds[0] for _ in range(size_diff)]

    return seeds


def calculate_and_save_metric_stats(
    config: types.RealConfig | types.PermutedConfig,
    read_directory: Path,
    save_directory: Path,
) -> None:
    var_config = config["turning_point"]

    if not var_config["should_calculate_it"]:
        return

    metrics = utils.parse_value_or_iterable(var_config["metric"])

    quantiles = utils.parse_value_or_iterable(var_config["quantile"])
    seeds = utils.parse_value_or_iterable(var_config["seed"])
    seeds = _extend_seeds_as_quantiles(quantiles, seeds)

    for metric in metrics:
        for seed, quantile in zip(seeds, quantiles):
            random.seed(seed)
            nprandom.seed(seed)

            fn_kwargs = {
                "var_parameters": config["turning_point"]["parameters"],
                "quantile": quantile,
                "metric": metric,
            }
            filename_to_var_stats = utils.run_for_all_filenames(
                _calculate_metric_stats,
                config["sports"],
                read_directory,
                **fn_kwargs,
            )

            save_dir = save_directory / str(quantile) / metric
            utils.save_filename_to_df(filename_to_var_stats, save_dir)
