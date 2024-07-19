from typing import Callable, Sequence

from tournament_simulations.schedules import Round

from ..algorithm import OptimalFn

SchedulingFn = Callable[
    [
        Sequence[str],  # team names
        int,  # num schedules
        str,  # second portion (turn) type
        OptimalFn,  # scheduling function to create an optimal single round-robin schedule
    ],
    list[Round],
]
