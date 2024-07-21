import random
from functools import partial
from pathlib import Path
from typing import Callable, Iterable

import numpy.random as nprandom
import pandas as pd

from logs import log, turning_logger
from synthetic_tournaments import Scheduler
from synthetic_tournaments.bradley_terry import simulate_bradley_terry_tourney
from synthetic_tournaments.optimal_schedule import algorithm as opt_alg
from synthetic_tournaments.optimal_schedule import scheduling as opt_sch
from synthetic_tournaments.optimal_schedule import scheduling as sch
from synthetic_tournaments.permutation import scheduling as sch
from tournament_simulations.data_structures import Matches
from tournament_simulations.permutations import MatchesPermutations

from .. import types
from . import utils


def _permute_matches(
    matches: Matches,
    permutation_fn: Callable,
    permutation_ids: Iterable[str],
) -> Matches:
    scheduler_factory = Scheduler(matches, permutation_fn)
    scheduler = scheduler_factory.get_current_year_scheduler()

    permutations_creator = MatchesPermutations(matches, scheduler)
    return permutations_creator.create_n_permutations(permutation_ids)


def _generate_permutations_one_result(
    label: str,
    strengths: list[float],
    number_of_drr: int,
    n_random_permutations: int,
) -> Matches:
    matches = simulate_bradley_terry_tourney(strengths, label, number_of_drr)

    random_kwargs = {
        "permutation_fn": sch.circle_method.create_double_rr,
        "permutation_ids": map(str, range(n_random_permutations)),
    }

    graph_kwargs = {
        "permutation_fn": partial(
            opt_sch.good_vs_bad_last.create_double_rr,
            second_portion="flipped",
            optimal_fn=opt_alg.generate_optimal_graph_schedule,
        ),
        "permutation_ids": ["graph_optimal"],
    }

    recursive_kwargs = {
        "permutation_fn": partial(
            opt_sch.good_vs_bad_last.create_double_rr,
            second_portion="flipped",
            optimal_fn=opt_alg.generate_recursive_optimal_schedule,
        ),
        "permutation_ids": ["recusive_optimal"],
    }

    all_kwargs = [random_kwargs, graph_kwargs, recursive_kwargs]
    to_concat = (_permute_matches(matches, **kwargs).df for kwargs in all_kwargs)
    return Matches(pd.concat(to_concat))


@log(turning_logger.info)
def _create_bt_permutations(
    label: str,
    strengths: list[float],
    n_different_results: int,
    n_permutations_per_result: int,
    number_of_drr: int,
) -> Matches:

    kwargs = {
        "strengths": strengths,
        "number_of_drr": number_of_drr,
        "n_random_permutations": n_permutations_per_result,
    }
    diff_permutations = (
        _generate_permutations_one_result(f"{label}@result_{i}", **kwargs)
        for i in range(n_different_results)
    )
    return Matches(pd.concat(matches.df for matches in diff_permutations))


def create_and_save_bradltey_terry_matches(
    config: types.BradleyTerryConfig,
    save_directory: Path,
) -> None:
    bt_cfg = config["matches"]
    if not bt_cfg["should_create_it"]:
        return

    random.seed(bt_cfg["seed"])
    nprandom.seed(bt_cfg["seed"])

    filenames = utils.parse_value_or_iterable(config["sports"])

    filename_to_matches = {
        filename: _create_bt_permutations("BT", **bt_cfg["parameters"][filename])
        for filename in filenames
    }

    utils.save_filename_to_df(filename_to_matches, save_directory)
