"""
Biggest strength difference (average per round) happens at the end.
"""

from typing import Sequence

import tournament_simulations.schedules.round_robin as rr
from tournament_simulations.schedules import Round
from tournament_simulations.schedules.utils.reversed_schedule import reverse_schedule

from ..algorithm import OptimalFn, generate_recursive_optimal_schedule
from .good_vs_bad_first import create_break_minimizing_double_rr


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
    drr.first_schedule = reverse_schedule(drr.first_schedule)
    drr.second_schedule = reverse_schedule(drr.second_schedule)
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
    drr.first_schedule = reverse_schedule(drr.first_schedule)
    drr.second_schedule = reverse_schedule(drr.second_schedule)
    to_randomize = ["home_away", "matches"]
    return list(drr.get_full_schedule(num_schedules, to_randomize, second_portion))


def create_break_minimizing_double_rr(
    team_names: Sequence[str],
    num_schedules: int,
    second_portion: str = "flipped",
    optimal_fn: OptimalFn = generate_recursive_optimal_schedule,
) -> list[Round]:

    params = {
        "team_names": team_names,
        "num_schedules": num_schedules,
        "second_portion": second_portion,
        "optimal_fn": optimal_fn,
    }
    minimizer_scheduele = create_break_minimizing_double_rr(**params)
    return list(reverse_schedule(minimizer_scheduele))
