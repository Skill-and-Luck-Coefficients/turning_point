import random
from functools import partial
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy.random as nprandom
import pandas as pd

from logs import log, turning_logger
from synthetic_tournaments import Scheduler
from synthetic_tournaments.optimal_schedule import algorithm as alg
from synthetic_tournaments.optimal_schedule import scheduling as sch
from tournament_simulations.data_structures import Matches
from tournament_simulations.permutations import MatchesPermutations
from tournament_simulations.schedules import Round

from .. import types
from . import utils

SimpleSchedulingFn = Callable[
    [
        Sequence[str],  # team names
        int,  # num schedules
    ],
    list[Round],
]


def _make_scheduling_fns(
    optimal_fn: alg.OptimalFn,
    deterministic_fn: sch.types.SchedulingFn,
    random_fn: sch.types.SchedulingFn,
) -> dict[types.ScheduleType, SimpleSchedulingFn]:
    fn_kwargs = {"optimal_fn": optimal_fn}

    return {
        "mirrored": partial(deterministic_fn, second_portion="flipped", **fn_kwargs),
        "reversed": partial(deterministic_fn, second_portion="reversed", **fn_kwargs),
        "random_mirrored": partial(random_fn, second_portion="flipped", **fn_kwargs),
        "random_reversed": partial(random_fn, second_portion="reversed", **fn_kwargs),
    }


SCHEDULING_FNS = {
    "graph": {
        "tp_maximizer": _make_scheduling_fns(
            alg.generate_optimal_graph_schedule,
            sch.good_vs_bad_last.create_double_rr,
            sch.good_vs_bad_last.create_random_double_rr,
        ),
        "tp_minimizer": _make_scheduling_fns(
            alg.generate_optimal_graph_schedule,
            sch.good_vs_bad_first.create_double_rr,
            sch.good_vs_bad_first.create_random_double_rr,
        ),
    },
    "recursive": {
        "tp_maximizer": _make_scheduling_fns(
            alg.generate_recursive_optimal_schedule,
            sch.good_vs_bad_last.create_double_rr,
            sch.good_vs_bad_last.create_random_double_rr,
        ),
        "tp_minimizer": _make_scheduling_fns(
            alg.generate_recursive_optimal_schedule,
            sch.good_vs_bad_first.create_double_rr,
            sch.good_vs_bad_first.create_random_double_rr,
        ),
    },
}


def _flat_iter_scheduling_fns_parameter(
    desired_types: types.OptimalMatchesTypeParameter,
) -> Iterator[tuple[str, SimpleSchedulingFn]]:
    for alg_key, tp_dict in desired_types.items():
        for tp_key, types in tp_dict.items():
            if isinstance(types, str):
                types = [types]

            for type_ in types:
                id_ = f"{alg_key}_{tp_key}_{type_}"
                fn = SCHEDULING_FNS[alg_key][tp_key][type_]
                yield id_, fn


def _concat_optimal_schedules_for_all_types(
    matches: Matches,
    desired_types: types.OptimalMatchesTypeParameter,
) -> Matches:
    all_tournaments: list[pd.DataFrame] = []

    for id_, scheduling_fn in _flat_iter_scheduling_fns_parameter(desired_types):
        scheduler_factory = Scheduler(matches, scheduling_fn)
        schedulers = {
            "current": scheduler_factory.get_current_year_scheduler(),
            "previous": scheduler_factory.get_previous_year_scheduler(),
        }

        for scheduler_type, scheduler in schedulers.items():
            # Need to filter since not all seasons have a 'previous_year'
            filtered_matches = Matches(matches.df.loc[scheduler.id_to_parameters.index])
            permutations_creator = MatchesPermutations(filtered_matches, scheduler)

            id_ = f"{id_}-{scheduler_type}"
            permuted_df = permutations_creator.create_n_permutations([id_]).df
            all_tournaments.append(permuted_df)

    return Matches(pd.concat(all_tournaments))


@log(turning_logger.info)
def _create_synthetic_matches(
    filepath: Path,
    desired_types: types.OptimalMatchesTypeParameter,
) -> Matches:
    matches = Matches(pd.read_csv(filepath))
    return _concat_optimal_schedules_for_all_types(matches, desired_types)


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

    fn_kwargs = {"desired_types": optimal_cfg["parameters"]["types"]}
    filename_to_matches = utils.run_for_all_filenames(
        _create_synthetic_matches,
        config["sports"],
        read_directory,
        **fn_kwargs,
    )

    utils.save_filename_to_df(filename_to_matches, save_directory)
