from typing import Callable, Literal

import numpy as np
import pandas as pd

import tournament_simulations.schedules.round_robin as ts_rr
from tournament_simulations.data_structures import Matches
from tournament_simulations.schedules import convert_list_of_rounds_to_dataframe


def _replace_team_name_with_strength(
    matches_df: pd.DataFrame,
    strengths: list[float],
) -> pd.DataFrame:

    strengths_ = dict(enumerate(strengths))
    return {
        "home": matches_df["home"].map(strengths_),
        "away": matches_df["away"].map(strengths_),
    }


def _get_bradley_terry_winner(
    skill_per_match: dict[Literal["home", "away"], pd.Series],
    uniform_values: np.ndarray,
) -> pd.Series:

    skill_per_match_sum = skill_per_match["home"] + skill_per_match["away"]
    prob_home_win = skill_per_match["home"] / skill_per_match_sum

    return (uniform_values <= prob_home_win).map({True: "h", False: "a"})


def _simulate_bt_tourney_no_randomness(
    strengths: list[float],
    label: str,
    number_of_drr: int,
    random_fn: Callable[[int], np.ndarray],
    rand_first: str | None,
) -> Matches:
    """
    random_fn: Callable[
        [int],      # number of elements (size)
        np.ndarray, # random uniform numbers
    ]
    """
    drr = ts_rr.DoubleRoundRobin.from_num_teams(len(strengths))
    schedule = drr.get_full_schedule(number_of_drr, rand_first)
    matches_df = convert_list_of_rounds_to_dataframe(schedule, label)

    uniform_values = random_fn(size=len(matches_df))
    skill_per_match = _replace_team_name_with_strength(matches_df, strengths)
    matches_df["winner"] = _get_bradley_terry_winner(skill_per_match, uniform_values)
    return Matches(matches_df)


def simulate_bradley_terry_tourney(
    strengths: list[float],
    label: str = "bradley_terry",
    number_of_drr: int = 1,
) -> Matches:
    """
    Simulate one tournament from Bradley-Terry's pairwise comparison probabilities.

    ----
    Parameters:
        strengths: list[float]
            Team strengths

        label: str = "bradley_terry"
            Tournament id

        number_of_drr: int = 1
            How many double round-robin should be concatenated together to create the tournament.

    ----
    Returns:
        Matches
            Tournament schedule.
    """
    return _simulate_bt_tourney_no_randomness(
        strengths,
        label,
        number_of_drr,
        random_fn=np.random.random,
        rand_first="all",
    )
