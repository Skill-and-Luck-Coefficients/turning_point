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


def _get_probability_per_id(
    winner_to_probability: dict | None,
    matches: Matches,
    winner_type: dict,
    winner_to_points: dict,
):
    def _from_config():
        _points_to_prob = {
            winner_to_points[_winner]: _probability
            for _winner, _probability in winner_to_probability.items()
        }
        _ids = matches.df.index.get_level_values("id").unique()
        return pd.Series(index=_ids, data=[_points_to_prob] * len(_ids)).sort_index()

    def _from_empirical_frequencies():
        _point_pairs = sorted(set(winner_to_points.values()))
        _ppm = PointsPerMatch.from_home_away_winner(
            home_away_winner=matches.home_away_winner(winner_type),
            result_to_points=winner_to_points,
        )
        return _ppm.probabilities_per_id(_point_pairs)

    if winner_to_probability is not None:
        return _from_config()

    return _from_empirical_frequencies()


def _get_metric_stats(
    matches: Matches,
    quantile: float,
    metric: str,
    **kwargs,
) -> ms.ExpandingMetricStats:
    """
    This function works both for real matches and permutation matches.

    For permutation matches it calculates stats for each permutation
    separately to reduce memory usage.
    """
    winner_to_points = {k: tuple(v) for k, v in kwargs["winner_to_points"].items()}

    all_var_stats: list[pd.DataFrame] = []
    permutation_ids = pc.get_permutation_identifiers(matches.df)

    for perm_id in permutation_ids:
        turning_logger.info(f"Starting i-th permutation: {perm_id}")
        filtered_matches = Matches(pc.get_data_with_identifier(matches.df, perm_id))

        params = {
            "winner_to_probability": kwargs.get("winner_to_probability"),
            "matches": filtered_matches,
            "winner_type": kwargs["winner_type"],
            "winner_to_points": winner_to_points,
        }
        id_to_probabilities = _get_probability_per_id(**params)

        params = {
            "matches": filtered_matches,
            "num_iteration_simulation": kwargs["num_iteration_simulation"],
            "winner_type": kwargs["winner_type"],
            "winner_to_points": winner_to_points,
            "id_to_probabilities": id_to_probabilities,
            "quantile": quantile,
            "metric_type": METRIC_MAP[metric],
        }
        var_stats = ms.ExpandingMetricStats.from_matches(**params)

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
