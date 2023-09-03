from dataclasses import dataclass
from typing import Any, Callable

from tournament_simulations.data_structures import Matches
from tournament_simulations.permutations import TournamentScheduler
from tournament_simulations.schedules import Round

from .num_schedules import from_matches
from .team_names import from_current_rankings
from .utils import agg_tuple_per_id


@dataclass
class Scheduler:
    """
    Tournament scheduler creator.

    matches: Matches
        Tournament matches.

    func: Callable[[list[Any], int], list[Round]]
        Scheduling function.
    """

    matches: Matches
    func: Callable[[list[Any], int], list[Round]]

    def get_current_year_scheduler(self) -> TournamentScheduler:
        """
        Create TournamentScheduler using current rankings.
        """
        id_to_team_names = from_current_rankings(self.matches)
        id_to_num_schedules = from_matches(self.matches)

        to_agg = [id_to_team_names, id_to_num_schedules]
        id_to_parameters = agg_tuple_per_id(to_agg)

        return TournamentScheduler(self.func, id_to_parameters)
