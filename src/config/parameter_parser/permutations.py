import random
from pathlib import Path

import numpy.random as nprandom
import pandas as pd

from logs import log, turning_logger
from synthetic_tournaments import Scheduler
from synthetic_tournaments.permutation import scheduling as sch
from tournament_simulations.data_structures import Matches
from tournament_simulations.permutations import MatchesPermutations

from .. import types
from . import utils


@log(turning_logger.info)
def _create_synthetic_matches(
    filepath: Path,
    permuted_parameters: types.PermutedMatchesParameters,
) -> Matches:
    matches = Matches(pd.read_csv(filepath))

    scheduler_factory = Scheduler(matches, sch.circle_method.create_double_rr)
    scheduler = scheduler_factory.get_current_year_scheduler()

    num_permutations = permuted_parameters["num_permutations"]
    permutations_creator = MatchesPermutations(matches, scheduler)
    return permutations_creator.create_n_permutations(num_permutations)


def create_and_save_permuted_matches(
    config: types.PermutedConfig,
    read_directory: Path,
    save_directory: Path,
) -> None:
    permuted_config = config["matches"]

    if not permuted_config["should_create_it"]:
        return

    random.seed(permuted_config["seed"])
    nprandom.seed(permuted_config["seed"])

    fn_kwargs = {"permuted_parameters": permuted_config["parameters"]}
    filename_to_matches = utils.run_for_all_filenames(
        _create_synthetic_matches,
        config["sports"],
        read_directory,
        **fn_kwargs,
    )

    utils.save_filename_to_df(filename_to_matches, save_directory)
