from typing import Callable, Sequence

from tournament_simulations.schedules import Round

OptimalFn = Callable[
    [int | Sequence[float]],  # number of teams | team strenghts/rankings
    list[Round],  # schedule
]
