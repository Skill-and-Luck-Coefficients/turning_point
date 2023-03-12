from typing import Literal

import pandas as pd

from logs import log, turning_logger
from tournament_simulations.data_structures import PointsPerMatch
from tournament_simulations.simulations import SimulatePointsPerMatch

KwargsVar = dict[Literal["real", "simulated"], pd.DataFrame]


def _calculate_ranking_variances_per_id(df: pd.DataFrame) -> pd.DataFrame:

    rankings = df.groupby(["id", "team"], observed=True).sum()
    return rankings.groupby("id", observed=True).var()


@log(turning_logger.debug)
def get_kwargs_from_points_per_match(
    ppm: PointsPerMatch,
    num_iteration_simulation: tuple[int, int],
) -> KwargsVar:

    """
    Simulates all tournaments and returns its variances.

    -----
    Parameters:

        points_per_match: PointsPerMatch
            Points each team gained in each match they played.

        num_iteration_simulation: tuple[int, int]
            Respectively, number of iterations and number
            of simulations per iteration (batch size).

    -----
    Returns:
        Kwargs parameters to create an instance of Variance
            "real": Ranking variance for all real tournaments.
            "simulated": Ranking variance for all simulated tournaments.
    """

    real_variances = _calculate_ranking_variances_per_id(ppm.df)

    simul_ppm = SimulatePointsPerMatch(ppm)
    simulated_variances = simul_ppm.tournament_wide(
        num_iteration_simulation=num_iteration_simulation,
        func_after_simulation=_calculate_ranking_variances_per_id,
    )

    return {
        "real": real_variances.rename(columns={"points": "real var"}),
        "simulated": simulated_variances,
    }
