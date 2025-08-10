from typing import Callable, Literal

import numpy as np
import pandas as pd

import tournament_simulations.schedules.round_robin as ts_rr
from tournament_simulations.data_structures import Matches
from tournament_simulations.schedules import convert_list_of_rounds_to_dataframe

HomeAdvantageKeys = Literal["additive", "multiplicative"]
NO_HOME_ADVANTAGE = {"additive": 0, "multiplicative": 1}


def _simulate_bt_tourney_no_randomness(
    strengths: list[float],
    label: str,
    number_of_drr: int,
    rand_first: str | None,
    scheduling_func: str | Callable,
    scheduling_func_type: Literal["rankings", "strengths"],
    home_advantage: dict[HomeAdvantageKeys, float],
    random_fn: Callable[[int], np.ndarray],
) -> Matches:
    """
    random_fn: Callable[
        [int],      # number of elements (size)
        np.ndarray, # random uniform numbers
    ]
    """

    def _get_ddr_scheduler():
        if scheduling_func_type == "rankings":
            _drr_args = (len(strengths), scheduling_func)
            return ts_rr.DoubleRoundRobin.from_num_teams(*_drr_args)

        _drr_kwargs = {
            "num_teams": len(strengths),
            "team_names": list(range(len(strengths))),
            "first_schedule": scheduling_func(strengths),
        }
        return ts_rr.DoubleRoundRobin(**_drr_kwargs)

    def _get_full_schedule_as_df() -> pd.DataFrame:
        _schedule = drr_scheduler.get_full_schedule(number_of_drr, rand_first)
        return convert_list_of_rounds_to_dataframe(_schedule, label)

    def _get_team_strengths_rowwise() -> pd.DataFrame:
        _strengths = dict(enumerate(strengths))
        return pd.DataFrame(
            {
                "home": schedule["home"].map(_strengths),
                "away": schedule["away"].map(_strengths),
            }
        )

    def _add_home_advantage(_skill_home: float) -> float:
        _skill_home = _skill_home + home_advantage["additive"]
        return home_advantage["multiplicative"] * _skill_home

    def _simulate_row_winner() -> pd.Series:
        _skill_home = _add_home_advantage(skill_per_match["home"])
        _skill_per_match_sum = _skill_home + skill_per_match["away"]
        _prob_home_win = _skill_home / _skill_per_match_sum

        _uniform_values = random_fn(len(schedule))
        return (_uniform_values <= _prob_home_win).map({True: "h", False: "a"})

    drr_scheduler = _get_ddr_scheduler()
    schedule = _get_full_schedule_as_df()

    skill_per_match = _get_team_strengths_rowwise()
    schedule["winner"] = _simulate_row_winner()

    return Matches(schedule)


def simulate_bradley_terry_tourney(
    strengths: list[float],
    label: str = "bradley_terry",
    number_of_drr: int = 1,
    scheduling_func: str | Callable = "circle",
    scheduling_func_type: Literal["rankings", "strengths"] = "rankings",
    randomize_schedule: str | list[str] | None = "all",
    home_advantage: dict[HomeAdvantageKeys, float] = NO_HOME_ADVANTAGE,
) -> Matches:
    """
    Simulate one tournament from Bradley-Terry's pairwise comparison probabilities.

    Let 'h' represent the home-team, and 'a' represent the away-team. <br>
    Then the probability that the home-team wins is given by:
    ```
        p_{ha} = strength_h / (strength_h + strength_a )
    ```

    ----
    Parameters:
        strengths: list[float]
            Team strengths

        label: str = "bradley_terry"
            Tournament id

        number_of_drr: int = 1
            How many double round-robin should be concatenated together to create the tournament.

        randomize_schedule: str | list[str] | None
            What should be randomized in the first portion of the double round-robin schedule.
                Options: ["teams", "home_away", "matches", "rounds", "all"]

                If it is an empty iterable or None, a copy of schedule will be returned.

        scheduling_func: str | Callable[[int], list[Round]] = "circle"
            Function responsible for creating a schedule.

        scheduling_func_type: Literal["rankings", "strenghts"] = "rankings"
            How to apply the scheduling function: to the rankings, or to the strengths.

        home_advantage: dict[Literal["additive", "multiplicative"], float] = NO_HOME_ADVANTAGE
            Add home advantage to the simulations.
            ```
            "additive": Flat value added to home-team strength.
            "multiplicative": Multiplies home-team strength.

            new_strength_h = (additive + strength_h) * multiplicative
            ```
    ----
    Returns:
        Matches
            Tournament schedule.
    """
    return _simulate_bt_tourney_no_randomness(
        strengths,
        label,
        number_of_drr,
        rand_first=randomize_schedule,
        scheduling_func=scheduling_func,
        scheduling_func_type=scheduling_func_type,
        home_advantage=home_advantage,
        random_fn=np.random.random,
    )
