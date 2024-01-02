from typing import Literal, Mapping

import pandas as pd

from tournament_simulations.data_structures import Matches, PointsPerMatch
from turning_point.variances import Variances

KwargsSVS = dict[Literal["df"], pd.DataFrame]


def _calculate_mean_quantile(
    simul_var_df: pd.DataFrame, quantile: float
) -> pd.DataFrame:
    quantiles = simul_var_df.quantile(quantile, axis=1).rename("0.950-quantile")
    mean = simul_var_df.mean(axis=1).rename("mean")

    return pd.concat([mean, quantiles], axis=1)


def get_kwargs_from_variances(variances: Variances, quantile: float) -> KwargsSVS:
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

    simulated_stats = _calculate_mean_quantile(variances.simulated, quantile)
    return {"df": pd.concat([variances.real, simulated_stats], axis=1)}


def get_kwargs_from_matches(
    matches: Matches,
    num_iteration_simulation: tuple[int, int],
    winner_type: Literal["winner", "result"],
    winner_to_points: Mapping[str, tuple[float, float]],
    id_to_probabilities: pd.Series | None = None,
    quantile: float = 0.95,
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

        id_to_probabilities: pd.Series | None = None
            Series mapping each tournament to its estimated probabilities.

            Probabilities:  Mapping[tuple[float, float]: float]
                Maps each pair (tuple) to its probability (float).

                Pair: ranking points gained respectively by home-team and away-team.

            If None, they will be estimated directly from 'matches'.

        winner_type: Literal["winner", "result"]
            What should points be based on.
                match: winner of the match
                    home: "h"
                    draw: "d"
                    away: "a"
                result: result of match: f{score home team}-{score away team}"

        winner_to_points: Mapping[str, tuple[float, float]]
            Mapping winner/result to how many points each team should gain.

            First tuple result is for home-team, while the second one is for away-team.
    ----
    Returns:

        Kwargs to create an instance of SimulationVarStats
            "df": DataFrame with real ranking-variance and mean/0.950-quantile
                  for ranking-variances in simulations.
    """
    ppm = PointsPerMatch.from_home_away_winner(
        home_away_winner=matches.home_away_winner(winner_type),
        result_to_points=winner_to_points,
    )

    if id_to_probabilities is not None:
        desired_ids = ppm.df.index.get_level_values("id").unique().sort_values()
        id_to_probabilities = id_to_probabilities.loc[desired_ids]

    variances = Variances.from_points_per_match(
        ppm, num_iteration_simulation, id_to_probabilities
    )

    simulated_stats = _calculate_mean_quantile(variances.simulated, quantile)
    return {"df": pd.concat([variances.real, simulated_stats], axis=1)}
