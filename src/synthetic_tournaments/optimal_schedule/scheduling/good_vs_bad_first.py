"""
Biggest strength difference (average per round) happens at the beginning.
"""

from typing import Sequence

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
