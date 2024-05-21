import random
from pathlib import Path

import numpy.random as nprandom
import pandas as pd

from logs import log, turning_logger
from synthetic_tournaments import Scheduler
from synthetic_tournaments.optimal_schedule import scheduling as sch
from tournament_simulations.data_structures import Matches
from tournament_simulations.permutations import MatchesPermutations

from .. import types

KEY_TO_SCHEDULING_FUNCTION = {
    "tp_minimizer": sch.good_vs_bad_first.create_double_rr,
    "tp_minimizer_reversed": sch.good_vs_bad_first.create_reversed_double_rr,
    "tp_maximizer": sch.good_vs_bad_last.create_double_rr,
    "tp_maximizer_reversed": sch.good_vs_bad_last.create_reversed_double_rr,
    "tp_minimizer_random": sch.good_vs_bad_first.create_random_double_rr,
    "tp_minimizer_random_reversed": sch.good_vs_bad_first.create_random_reversed_double_rr,
    "tp_maximizer_random": sch.good_vs_bad_last.create_random_double_rr,
    "tp_maximizer_random_reversed": sch.good_vs_bad_last.create_random_reversed_double_rr,
}


def _concat_optimal_schedules_for_all_types(
    matches: Matches,
    desired_types: types.OptimalScheduleTypes | list[types.OptimalScheduleTypes],
) -> Matches:
    if isinstance(desired_types, str):
        desired_types = [desired_types]

    all_tournaments: list[pd.DataFrame] = []

    for type_ in desired_types:
        scheduler_factory = Scheduler(matches, sch.KEY_TO_SCHEDULING_FUNCTION[type_])
        schedulers = {
            "current": scheduler_factory.get_current_year_scheduler(),
            "previous": scheduler_factory.get_previous_year_scheduler(),
        }

        for scheduler_type, scheduler in schedulers.items():
            # Need to filter since not all seasons have a 'previous_year'
            filtered_matches = Matches(matches.df.loc[scheduler.id_to_parameters.index])
            permutations_creator = MatchesPermutations(filtered_matches, scheduler)

            id_ = f"{type_}-{scheduler_type}"
            permuted_df = permutations_creator.create_n_permutations([id_]).df
            all_tournaments.append(permuted_df)

    return Matches(pd.concat(all_tournaments))


@log(turning_logger.info)
def _create_synthetic_matches(
    filenames: list[str],
    read_directory: Path,
    desired_types: types.OptimalScheduleTypes | list[types.OptimalScheduleTypes],
) -> dict[str, Matches]:
    filename_to_matches = {}

    for filename in filenames:
        filepath = read_directory / f"{filename}.csv"
        if not filepath.exists():
            turning_logger.warning(f"No file: {filepath}")
            continue

        matches = Matches(pd.read_csv(filepath))
        optimal_matches = _concat_optimal_schedules_for_all_types(
            matches, desired_types
        )
        filename_to_matches[filename] = optimal_matches

    return filename_to_matches


def create_and_save_optimal_matches(
    config: types.OptimalConfig,
    read_directory: Path,
    save_directory: Path,
) -> None:
    optimal_cfg = config["matches"]
    if not optimal_cfg["should_create_it"]:
        return

    random.seed(optimal_cfg["seed"])
    nprandom.seed(optimal_cfg["seed"])

    filename_to_matches = _create_synthetic_matches(
        config["sports"],
        read_directory,
        optimal_cfg["parameters"]["types"],
    )

    save_directory.mkdir(parents=True, exist_ok=True)
    for filename, matches in filename_to_matches.items():
        matches.df.to_csv(save_directory / f"{filename}.csv")
