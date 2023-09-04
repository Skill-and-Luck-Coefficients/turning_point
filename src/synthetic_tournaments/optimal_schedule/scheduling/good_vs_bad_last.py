"""
Biggest strength difference (average per round) happens at the end.
"""
from typing import Sequence

import tournament_simulations.schedules.round_robin as rr
from tournament_simulations.schedules import Round

from ..algorithm import generate_optimal_schedule


def create_double_rr(team_names: Sequence[str], num_schedules: int) -> list[Round]:
    """
    Symmetric schedule: second portion is the first one with
    (home, away) matches as (away, home).
    """
    drr = rr.DoubleRoundRobin.from_team_names(team_names, generate_optimal_schedule)
    schedule = list(drr.get_full_schedule(num_schedules, None, "flipped"))
    return [tuple(reversed(round_)) for round_ in reversed(schedule)]


def create_random_double_rr(
    team_names: Sequence[str], num_schedules: int
) -> list[Round]:
    """
    Symmetric schedule: second portion is the first one with
    (home, away) matches as (away, home).

    Randomizes which team play as home/away.
    """
    drr = rr.DoubleRoundRobin.from_team_names(team_names, generate_optimal_schedule)
    schedule = list(drr.get_full_schedule(num_schedules, "home_away", "flipped"))
    return [tuple(reversed(round_)) for round_ in reversed(schedule)]
