"""
Biggest strength difference (average per round) happens at the beginning.
"""

from typing import Sequence

import synthetic_tournaments.break_minimizer.scheduling as min_break
import tournament_simulations.schedules.round_robin as rr
from tournament_simulations.schedules import Round

from ..algorithm import OptimalFn, generate_recursive_optimal_schedule


def create_double_rr(
    team_names: Sequence[str],
    num_schedules: int,
    second_portion: str = "flipped",
    optimal_fn: OptimalFn = generate_recursive_optimal_schedule,
) -> list[Round]:
    """
    Symmetric schedule: second portion is the first one with
    (home, away) matches as (away, home).
    """
    drr = rr.DoubleRoundRobin.from_team_names(team_names, optimal_fn)
    return list(drr.get_full_schedule(num_schedules, None, second_portion))


def create_random_double_rr(
    team_names: Sequence[str],
    num_schedules: int,
    second_portion: str = "flipped",
    optimal_fn: OptimalFn = generate_recursive_optimal_schedule,
) -> list[Round]:
    """
    Symmetric schedule: second portion is the first one with
    (home, away) matches as (away, home).

    Randomizes which team play as home/away.
    """
    drr = rr.DoubleRoundRobin.from_team_names(team_names, optimal_fn)
    to_randomize = ["home_away", "matches"]
    return list(drr.get_full_schedule(num_schedules, to_randomize, second_portion))


def create_break_minimizing_double_rr(
    team_names: Sequence[str],
    num_schedules: int,
    second_portion: str = "flipped",
    optimal_fn: OptimalFn = generate_recursive_optimal_schedule,
) -> list[Round]:

    def _min_break_scheduling_fn(teams: int):
        optimal_schedule = optimal_fn(teams)
        return min_break.min_break_schedule_from_list_schedule(optimal_schedule)

    params = {
        "team_names": team_names,
        "num_schedules": num_schedules,
        "second_portion": second_portion,
        "optimal_fn": _min_break_scheduling_fn,
    }
    return create_double_rr(**params)
