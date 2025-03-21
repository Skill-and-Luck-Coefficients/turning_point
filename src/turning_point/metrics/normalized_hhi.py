from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from synthetic_tournaments.most_imbalanced import build_most_imbalanced_tournament
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


hhi_fn = herfindahl_hirschman_index


def hhi_lower_bound(standings: pd.Series):
    return 1 / standings.groupby("id", observed=True).size()


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

        def _calculate_normalized_hhi_per_id(df: pd.DataFrame) -> pd.DataFrame:
            def _get_upper_bound():
                _df = build_most_imbalanced_tournament(df)
                _rankings = _df.groupby(["id", "team"], observed=True).sum()
                return _rankings.groupby("id", observed=True).apply(hhi_fn)

            rankings = df.groupby(["id", "team"], observed=True).sum()

            lower_bound = hhi_lower_bound(rankings)
            upper_bound = _get_upper_bound()

            hhi = rankings.groupby("id", observed=True).apply(hhi_fn)
            nhhi = (hhi - lower_bound) / (upper_bound - lower_bound)
            return np.clip(nhhi, 0, 1)

        parameters = get_kwargs_from_points_per_match(
            ppm,
            _calculate_normalized_hhi_per_id,
            num_iteration_simulation,
            id_to_probabilities,
        )
        return cls(**parameters)


@dataclass
class NaiveNormalizedHHI(Metric):
    """
    Naive Herfindahl Hirschman Index (upper_bound = 1).

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

        def _calculate_naive_normalized_hhi_per_id(df: pd.DataFrame) -> pd.DataFrame:
            rankings = df.groupby(["id", "team"], observed=True).sum()

            lower_bound = hhi_lower_bound(rankings)

            hhi = rankings.groupby("id", observed=True).apply(hhi_fn)
            return (hhi - lower_bound) / (1 - lower_bound)

        parameters = get_kwargs_from_points_per_match(
            ppm,
            _calculate_naive_normalized_hhi_per_id,
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

        def _calculate_hicb_per_id(df: pd.DataFrame) -> pd.DataFrame:
            rankings = df.groupby(["id", "team"], observed=True).sum()

            lower_bound = hhi_lower_bound(rankings)

            hhi = rankings.groupby("id", observed=True).apply(hhi_fn)
            return hhi / lower_bound

        parameters = get_kwargs_from_points_per_match(
            ppm,
            _calculate_hicb_per_id,
            num_iteration_simulation,
            id_to_probabilities,
        )
        return cls(**parameters)
