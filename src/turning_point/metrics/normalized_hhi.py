from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tournament_simulations.data_structures import PointsPerMatch

from .calculate_metric import get_kwargs_from_points_per_match
from .metric import Metric


def herfindahl_hirschman_index(
    standings: pd.Series | pd.DataFrame,
) -> float | pd.Series:
    """
    standings:
        pd.DataFrame: Returns pd.Series for each column
        pd.Series: Returns float value
    """
    normalized_standings = standings / standings.sum()
    return (normalized_standings**2).sum()


def normalized_herfindahl_hirschman_index(
    standings: pd.Series | pd.DataFrame,
) -> float | pd.Series:
    """
    standings:
        pd.DataFrame: Returns pd.Series for each column
        pd.Series: Returns float value
    """
    # NOTE: not all of these coefficients work the best when we consider the the 3-0 point system
    hhi = herfindahl_hirschman_index(standings)
    inverse_num_teams = 1 / len(standings)
    return (hhi - inverse_num_teams) / (1 - inverse_num_teams)


def _calculate_normalized_hhi_per_id(df: pd.DataFrame) -> pd.DataFrame:
    rankings = df.groupby(["id", "team"], observed=True).sum()
    nnhi_fn = normalized_herfindahl_hirschman_index
    return rankings.groupby("id", observed=True).agg(nnhi_fn)


def herfindahl_index_of_competitive_balance(
    standings: pd.Series | pd.DataFrame,
) -> float | pd.Series:
    """
    standings:
        pd.DataFrame: Returns pd.Series for each column
        pd.Series: Returns float value
    """
    hhi = herfindahl_hirschman_index(standings)
    inverse_num_teams = 1 / len(standings)
    return hhi / inverse_num_teams


def _calculate_hicb_per_id(df: pd.DataFrame) -> pd.DataFrame:
    rankings = df.groupby(["id", "team"], observed=True).sum()
    hicb_fn = herfindahl_index_of_competitive_balance
    return rankings.groupby("id", observed=True).agg(hicb_fn)


@dataclass
class NormalizedHHI(Metric):
    """
    Normalized Herfindahl Hirschman Index.

        real:
            pd.DataFrame[
                index=[
                    "id" -> pd.Categorical[str]
                        "{current_name}@/{sport}/{country}/{name-year}/",
                ],\n
                columns=[
                    "real" -> np.float64
                        "real": NormalizedHHI for real tournaments),\n
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
                        f"s{i}"   : NormalizedHHI for i-th simulation,
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
    ) -> NormalizedHHI:
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
            _calculate_normalized_hhi_per_id,
            num_iteration_simulation,
            id_to_probabilities,
        )
        return cls(**parameters)


@dataclass
class HICB(Metric):
    """
    Herfindahl Index of Competitive Balance.

        real:
            pd.DataFrame[
                index=[
                    "id" -> pd.Categorical[str]
                        "{current_name}@/{sport}/{country}/{name-year}/",
                ],\n
                columns=[
                    "real" -> np.float64
                        "real": HICB for real tournaments),\n
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
                        f"s{i}"   : HICB for i-th simulation,
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
    ) -> HICB:
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
            ppm, _calculate_hicb_per_id, num_iteration_simulation, id_to_probabilities
        )
        return cls(**parameters)
