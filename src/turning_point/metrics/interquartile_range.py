from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tournament_simulations.data_structures import PointsPerMatch

from .calculate_metric import get_kwargs_from_points_per_match
from .metric import Metric


def interquartile_range(standings: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    standings:
        pd.DataFrame: Returns pd.Series for each column
        pd.Series: Returns float value
    """
    return standings.quantile(0.75) - standings.quantile(0.25)


def _calculate_interquartile_range_per_id(df: pd.DataFrame) -> pd.DataFrame:
    rankings = df.groupby(["id", "team"], observed=True).sum()
    iqr_fn = interquartile_range
    return rankings.groupby("id", observed=True).apply(iqr_fn)


@dataclass
class IQR(Metric):
    """
    Interquartile range.

        real:
            pd.DataFrame[
                index=[
                    "id" -> pd.Categorical[str]
                        "{current_name}@/{sport}/{country}/{name-year}/",
                ],\n
                columns=[
                    "real" -> np.float64
                        "real": interquartile range for real tournaments),\n
                ]
            ]

        simulation:
            pd.DataFrame[
                index=[
                    "id" -> pd.Categorical[str]
                        "{current_name}@/{sport}/{country}/{name-year}/",
                ],\n
                columns=[
                    f"s{i}"-> np.float64
                        f"s{i}"   : interquartile range for i-th simulation,
                ]
            ]
    """

    real: pd.DataFrame
    simulated: pd.DataFrame

    @classmethod
    def from_points_per_match(
        cls,
        ppm: PointsPerMatch,
        num_iteration_simulation: tuple[int, int],
        id_to_probabilities: pd.Series | None = None,
    ) -> IQR:
        """
        Creates an instance from PointsPerMatch.

        -----
        Parameters:

            points_per_match: PointsPerMatch
                Points each team gained in each match they played.

            num_iteration_simulation: tuple[int, int]
                Respectively, number of iterations and number
                of simulations per iteration (batch size).

            id_to_probabilities: pd.Series | None = None
                Series mapping each tournament to its estimated probabilities.

                Probabilities:  Mapping[tuple[float, float]: float]
                    Maps each pair (tuple) to its probability (float).

                    Pair: ranking points gained respectively by home-team and away-team.

                If None, they will be estimated directly from 'ppm'.
        """

        parameters = get_kwargs_from_points_per_match(
            ppm,
            _calculate_interquartile_range_per_id,
            num_iteration_simulation,
            id_to_probabilities,
        )
        return cls(**parameters)
