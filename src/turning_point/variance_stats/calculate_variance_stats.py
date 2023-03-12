from typing import Literal

import pandas as pd

from tournament_simulations.data_structures import Matches, PointsPerMatch
from turning_point.variances import Variances

KwargsSVS = dict[Literal["df"], pd.DataFrame]


def _calculate_mean_quantile(simul_var_df: pd.DataFrame) -> pd.DataFrame:

    quantile = simul_var_df.quantile(0.95, axis=1).rename("0.950-quantile")
    mean = simul_var_df.mean(axis=1).rename("mean")

    return pd.concat([mean, quantile], axis=1)


def get_kwargs_from_variances(variances: Variances) -> KwargsSVS:

    """
    Get Kwargs parameters to create an instance of SimulationVarStats

    ----
    Parameters:

        variances: Variances
            Ranking-variances for real and simulated tournaments.

    ----
    Returns:

        Kwargs to create an instance of SimulationVarStats
            "df": DataFrame with real ranking-variance and mean/0.950-quantile
                  for ranking-variances in simulations.
    """

    simulated_stats = _calculate_mean_quantile(variances.simulated)
    return {"df": pd.concat([variances.real, simulated_stats], axis=1)}


def get_kwargs_from_matches(
    matches: Matches, num_iteration_simulation: tuple[int, int]
) -> KwargsSVS:

    """
    Get Kwargs parameters to create an instance of SimulationVarStats

    ----
    Parameters:

        matches: Matches
            Tournament matches.

        num_iteration_simulation: tuple[int, int]
            Respectively, number of iterations and number of
            simulations per iteration (batch size).

    ----
    Returns:

        Kwargs to create an instance of SimulationVarStats
            "df": DataFrame with real ranking-variance and mean/0.950-quantile
                  for ranking-variances in simulations.
    """
    ppm = PointsPerMatch.from_home_away_winner(matches.home_away_winner)

    variances = Variances.from_points_per_match(ppm, num_iteration_simulation)

    simulated_stats = _calculate_mean_quantile(variances.simulated)
    return {"df": pd.concat([variances.real, simulated_stats], axis=1)}
