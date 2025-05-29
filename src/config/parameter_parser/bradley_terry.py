import random
from pathlib import Path
from typing import Literal

import numpy.random as nprandom
import pandas as pd

from logs import log, turning_logger
from synthetic_tournaments import Scheduler
from synthetic_tournaments.bradley_terry import simulate_bradley_terry_tourney
from synthetic_tournaments.break_minimizer import scheduling as min_break
from synthetic_tournaments.optimal_schedule import algorithm as opt_alg
from tournament_simulations.data_structures import Matches
from tournament_simulations.permutations import MatchesPermutations, TournamentScheduler
from tournament_simulations.schedules.round_robin import DoubleRoundRobin
from tournament_simulations.schedules.utils import reversed_schedule

from .. import types
from . import utils


def _set_schedule(
    matches: Matches,
    random_label: str,
    label: str,
    **kwargs,
) -> Matches:
    """
    Permute schedule with the given algorithm.
    """

    def _scheduling_fn(_team_names: list[str]):
        _scheduling_func = kwargs.get("scheduling_func")
        _number_of_drr = kwargs.get("number_of_drr")
        _drr = DoubleRoundRobin.from_num_teams(len(_team_names), _scheduling_func)
        return _drr.get_full_schedule(_number_of_drr, None)

    def _get_renamed_index() -> pd.MultiIndex:
        # fmt: off
        _new_index_arrays = [
            df.index.get_level_values("id").str.replace(random_label, label).str.slice(0, -1),
            df.index.get_level_values("date number"),
        ]
        # fmt: on
        return pd.MultiIndex.from_arrays(_new_index_arrays)

    scheduler_params = {
        "func_schedule": _scheduling_fn,
        "id_to_parameters": matches.team_names_per_id.apply(lambda t: [sorted(t)]),
    }
    scheduler = TournamentScheduler(**scheduler_params)

    permutations_creator = MatchesPermutations(matches, scheduler)
    df = permutations_creator.create_n_permutations(n=[""]).df
    return Matches(df.set_index(_get_renamed_index()))


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
            "label": f"{label}@{_simulation_number}",
        }
        return simulate_bradley_terry_tourney(**simulation_params).df

    to_concat = (_simulate(number) for number in range(n_simulations))
    return Matches(pd.concat(to_concat))


@log(turning_logger.info)
def _create_bt_simulations(
    label: str,
    strengths: list[float],
    n_simulations: int,
    number_of_drr: int,
    type_simulation: Literal["all_random", "same_results"] = "same_results",
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

    def _rec_optimal_params():
        return {
            "strengths": strengths,
            "label": f"{label}_recursive_optimal",
            "number_of_drr": number_of_drr,
            "scheduling_func": opt_alg.generate_recursive_optimal_schedule,
            "randomize_schedule": None,
        }

    def _graph_optimal_max_params():
        def _scheduling_func(_strenghts):
            optimal_schedule = opt_alg.generate_optimal_graph_schedule(_strenghts)
            return reversed_schedule.reverse_schedule(optimal_schedule)

        return {
            "strengths": strengths,
            "label": f"{label}_graph_optimal_max",
            "number_of_drr": number_of_drr,
            "scheduling_func": _scheduling_func,
            "randomize_schedule": None,
        }

    def _rec_optimal_max_params():
        def _scheduling_func(_strenghts):
            optimal_schedule = opt_alg.generate_optimal_graph_schedule(_strenghts)
            return reversed_schedule.reverse_schedule(optimal_schedule)

        return {
            "strengths": strengths,
            "label": f"{label}_recursive_optimal_max",
            "number_of_drr": number_of_drr,
            "scheduling_func": _scheduling_func,
            "randomize_schedule": None,
        }

    def _min_break_graph_optimal_params():
        def _scheduling_func(_strenghts):
            optimal_schedule = opt_alg.generate_optimal_graph_schedule(_strenghts)
            return min_break.min_break_schedule_from_list_schedule(optimal_schedule)

        return {
            "strengths": strengths,
            "label": f"{label}_min_break_graph_optimal",
            "number_of_drr": number_of_drr,
            "scheduling_func": _scheduling_func,
            "randomize_schedule": None,
        }

    def _min_break_rec_optimal_params():
        def _scheduling_func(_strenghts):
            optimal_schedule = opt_alg.generate_recursive_optimal_schedule(_strenghts)
            return min_break.min_break_schedule_from_list_schedule(optimal_schedule)

        return {
            "strengths": strengths,
            "label": f"{label}_min_break_recursive_optimal",
            "number_of_drr": number_of_drr,
            "scheduling_func": _scheduling_func,
            "randomize_schedule": None,
        }

    if type_simulation == "all_random":
        simulations = [
            _generate_bt_simulations(n_simulations, **_purely_random_params()).df,
            _generate_bt_simulations(n_simulations, **_graph_optimal_params()).df,
            _generate_bt_simulations(n_simulations, **_rec_optimal_params()).df,
            _generate_bt_simulations(n_simulations, **_graph_optimal_max_params()).df,
            _generate_bt_simulations(n_simulations, **_rec_optimal_max_params()).df,
        ]

    if type_simulation == "same_results":
        simulation_params = _purely_random_params()
        simulations = _generate_bt_simulations(n_simulations, **simulation_params)

        random_label = simulation_params.get("label")
        simulations = [
            simulations.df,
            _set_schedule(simulations, random_label, **_graph_optimal_params()).df,
            _set_schedule(simulations, random_label, **_rec_optimal_params()).df,
            _set_schedule(simulations, random_label, **_graph_optimal_max_params()).df,
            _set_schedule(simulations, random_label, **_rec_optimal_max_params()).df,
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
