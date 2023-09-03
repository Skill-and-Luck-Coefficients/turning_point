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

        scheduler_factory = Scheduler(matches, sch.circle_method.create_double_rr)
        scheduler = scheduler_factory.get_current_year_scheduler()

        permutations_creator = MatchesPermutations(matches, scheduler)

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
