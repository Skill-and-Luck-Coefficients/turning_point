import pandas as pd

from tournament_simulations.data_structures import Matches
from tournament_simulations.permutations import TournamentScheduler
from tournament_simulations.schedules import round_robin as rr


def _create_double_rr(team_names, num_schedules):
    """
    Circle algorithm.

    Symmetric schedule: second portion is the first one with
    (home, away) matches as (away, home).
    """
    drr = rr.DoubleRoundRobin.from_team_names(team_names, "circle")
    return list(drr.get_full_schedule(num_schedules, "all", "flipped"))


class PermutationScheduler:
    """
    Scheduler for tournament permutations.

    Used for PermutationTuningPoint.
    """

    @staticmethod
    def from_matches(matches: Matches) -> TournamentScheduler:
        """
        Create TournamentScheduler from matches.
        """
        match_count = matches.home_vs_away_count_per_id
        id_to_num_schedules = match_count.groupby("id", observed=True).max()

        parameter_series = [matches.team_names_per_id, id_to_num_schedules]
        id_to_parameters_df = pd.concat(parameter_series, axis="columns")
        id_to_parameters = id_to_parameters_df.agg(tuple, axis="columns")

        return TournamentScheduler(_create_double_rr, id_to_parameters)
