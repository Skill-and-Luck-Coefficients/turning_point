"""
Biggest strength difference (average per round) happens at the end.
"""
from functools import partial
from typing import Sequence

import tournament_simulations.schedules.round_robin as rr
from tournament_simulations.schedules import Round
from tournament_simulations.schedules.utils.reversed_schedule import reverse_schedule

from ..algorithm import generate_optimal_schedule


def _create_double_rr(
    team_names: Sequence[str], num_schedules: int, second_portion: str
) -> list[Round]:
    """
    Symmetric schedule: second portion is the first one with
    (home, away) matches as (away, home).
    """
    drr = rr.DoubleRoundRobin.from_team_names(team_names, generate_optimal_schedule)
    drr.first_schedule = reverse_schedule(drr.first_schedule)
    drr.second_schedule = reverse_schedule(drr.second_schedule)
    return list(drr.get_full_schedule(num_schedules, None, second_portion))


create_double_rr = partial(_create_double_rr, second_portion="flipped")
create_reversed_double_rr = partial(_create_double_rr, second_portion="reversed")


def _create_random_double_rr(
    team_names: Sequence[str], num_schedules: int, second_portion: str
) -> list[Round]:
    """
    Symmetric schedule: second portion is the first one with
    (home, away) matches as (away, home).

    Randomizes which team play as home/away.
    """
    drr = rr.DoubleRoundRobin.from_team_names(team_names, generate_optimal_schedule)
    drr.first_schedule = reverse_schedule(drr.first_schedule)
    drr.second_schedule = reverse_schedule(drr.second_schedule)
    to_randomize = ["home_away", "matches"]
    return list(drr.get_full_schedule(num_schedules, to_randomize, second_portion))


create_random_double_rr = partial(_create_random_double_rr, second_portion="flipped")
create_random_reversed_double_rr = partial(
    _create_random_double_rr, second_portion="reversed"
)
