from typing import Any, Callable, Sequence

from tournament_simulations.schedules import Round

OptimalFn = Callable[
    [int | Sequence[Any]],  # teams
    list[Round],  # schedule
]
