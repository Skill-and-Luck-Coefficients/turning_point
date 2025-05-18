"""
Biggest strength difference (average per round) happens at the beginning.
"""

from typing import Sequence

import synthetic_tournaments.break_minimizer.scheduling as min_break
import tournament_simulations.schedules.round_robin as rr
from tournament_simulations.schedules import Round, rename_teams_in_rounds

from ..algorithm import OptimalFn, generate_recursive_optimal_schedule

MIN_BREAKS_CACHE = {}


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

    def _build_schedule_to_cache():
        params = {
            "team_names": list(range(num_teams)),
            "num_schedules": 1,
            "second_portion": second_portion,
            "optimal_fn": _min_break_scheduling_fn,
        }
        return create_double_rr(**params)

    num_teams = len(team_names)
    key = f"{optimal_fn.__name__}_{num_teams}"

    if key not in MIN_BREAKS_CACHE:
        MIN_BREAKS_CACHE[key] = _build_schedule_to_cache()

    schedule = MIN_BREAKS_CACHE[key] * num_schedules
    return list(rename_teams_in_rounds(schedule, team_names))
