from typing import Sequence

from tournament_simulations.schedules import Round
from tournament_simulations.schedules import round_robin as rr


def create_double_rr(team_names: Sequence[str], num_schedules: int) -> list[Round]:
    """
    Circle algorithm.

    Symmetric schedule: second portion is the first one with
    (home, away) matches as (away, home).
    """
    drr = rr.DoubleRoundRobin.from_team_names(team_names, "circle")
    return list(drr.get_full_schedule(num_schedules, "all", "flipped"))
