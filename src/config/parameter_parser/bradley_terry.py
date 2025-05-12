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
from synthetic_tournaments.permutation import scheduling as sch
from tournament_simulations.data_structures import Matches
from tournament_simulations.permutations import MatchesPermutations
from tournament_simulations.schedules import Round

from .. import types
from . import utils


def _generate_bt_simulations(
    n_simulations: int,
    label: str,
    **simulate_bt_kwargs,
) -> Matches:
    """
    Random schedule and random results.
    """

    def _simulate(_simulation_number: int) -> pd.DataFrame:
        simulation_params = {
            **simulate_bt_kwargs,
            "label": f"{label}_{_simulation_number}",
        }
        return simulate_bradley_terry_tourney(**simulation_params).df

    to_concat = (_simulate(number) for number in n_simulations)
    return Matches(pd.concat(to_concat))


@log(turning_logger.info)
def _create_bt_simulations(
    label: str,
    strengths: list[float],
    n_simulations: int,
    number_of_drr: int,
) -> Matches:

    def _purely_random_params():
        return {
            "strengths": strengths,
            "label": f"{label}_purely_random",
            "number_of_drr": number_of_drr,
            "scheduling_func": "circle",
            "randomize_schedule": "all",
        }

    def _graph_optimal_params():
        return {
            "strengths": strengths,
            "label": f"{label}_graph_optimal",
            "number_of_drr": number_of_drr,
            "scheduling_func": opt_alg.generate_optimal_graph_schedule,
            "randomize_schedule": None,
        }

    def _recursive_optimal_params():
        return {
            "strengths": strengths,
            "label": f"{label}_recursive_optimal",
            "number_of_drr": number_of_drr,
            "scheduling_func": opt_alg.generate_recursive_optimal_schedule,
            "randomize_schedule": None,
        }

    simulations = [
        _generate_bt_simulations(n_simulations, **_purely_random_params()).df,
        _generate_bt_simulations(n_simulations, **_graph_optimal_params()).df,
        _generate_bt_simulations(n_simulations, **_recursive_optimal_params()).df,
    ]
    return Matches(pd.concat(simulations))


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
        filename: _create_bt_simulations("BT", **bt_cfg["parameters"][filename])
        for filename in filenames
    }

    utils.save_filename_to_df(filename_to_matches, save_directory)
