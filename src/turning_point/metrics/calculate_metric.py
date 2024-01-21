from typing import Callable, Literal

import pandas as pd

from logs import log, turning_logger
from tournament_simulations.data_structures import PointsPerMatch
from tournament_simulations.simulations import SimulatePointsPerMatch

KwargsVar = dict[Literal["real", "simulated"], pd.DataFrame]


@log(turning_logger.debug)
def get_kwargs_from_points_per_match(
    ppm: PointsPerMatch,
    func: Callable[[pd.DataFrame], pd.DataFrame],
    num_iteration_simulation: tuple[int, int],
    id_to_probabilities: pd.Series | None = None,
) -> KwargsVar:
    """
    Simulates all tournaments and returns its variances.

    -----
    Parameters:

        points_per_match: PointsPerMatch
            Points each team gained in each match they played.

        func: Callable[[pd.DataFrame], pd.DataFrame]
            Function to calculate the desired metric.

            Input: PointPerMatch (df)
            Output: pd.DataFrame[
                index = [
                    "id" -> pd.Categorical[str]
                        "{current_name}@/{sport}/{country}/{name-year}/",
                ]
                columns = [
                    "points": float
                        Metric value
                ]
            ]

        num_iteration_simulation: tuple[int, int]
            Respectively, number of iterations and number
            of simulations per iteration (batch size).

        id_to_probabilities: pd.Series | None = None
            Series mapping each tournament to its estimated probabilities.

            Probabilities:  Mapping[tuple[float, float]: float]
                Maps each pair (tuple) to its probability (float).

                Pair: ranking points gained respectively by home-team and away-team.

            If None, they will be estimated directly from 'ppm'.

    -----
    Returns:
        Kwargs parameters to create an instance of Variance
            "real": Ranking variance for all real tournaments.
            "simulated": Ranking variance for all simulated tournaments.
    """

    real_variances = func(ppm.df)

    simul_ppm = SimulatePointsPerMatch(ppm)
    simulated_variances = simul_ppm.tournament_wide(
        num_iteration_simulation=num_iteration_simulation,
        id_to_probabilities=id_to_probabilities,
        func_after_simulation=func,
    )

    return {
        "real": real_variances.rename(columns={"points": "real"}),
        "simulated": simulated_variances,
    }
