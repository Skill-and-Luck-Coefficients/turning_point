from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from synthetic_tournaments.most_imbalanced import build_most_imbalanced_tournament
from tournament_simulations.data_structures import PointsPerMatch

from .calculate_metric import get_kwargs_from_points_per_match
from .metric import Metric


def gini(standings: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    standings:
        pd.DataFrame: Returns pd.Series for each column
        pd.Series: Returns float value
    """

    def _df_iterator(standings: pd.DataFrame) -> Iterator[pd.Series]:
        for _, value in standings.iterrows():
            yield value

    num_teams = len(standings)
    mean = standings.mean()

    sum_abs_diff = 0

    iterator = standings
    if isinstance(standings, pd.DataFrame):
        iterator = _df_iterator(standings)

    for value in iterator:
        value_abs_diff = (standings - value).abs().sum()
        sum_abs_diff += value_abs_diff

    return sum_abs_diff / (2 * mean * num_teams**2)


@dataclass
class Gini(Metric):
    """
    Gini Index.

        real:
            pd.DataFrame[
                index=[
                    "id" -> pd.Categorical[str]
                        "{current_name}@/{sport}/{country}/{name-year}/",
                ],\n
                columns=[
                    "real" -> np.float64
                        "real": gini index for real tournaments),\n
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
                        f"s{i}"   : gini index for i-th simulation,
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
    ) -> Gini:
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

        def _calculate_gini_index_per_id(df: pd.DataFrame) -> pd.DataFrame:
            rankings = df.groupby(["id", "team"], observed=True).sum()
            return rankings.groupby("id", observed=True).apply(gini)

        parameters = get_kwargs_from_points_per_match(
            ppm,
            _calculate_gini_index_per_id,
            num_iteration_simulation,
            id_to_probabilities,
        )
        return cls(**parameters)


@dataclass
class NormalizedGini(Metric):
    """
    Normalized Gini Index.

        real:
            pd.DataFrame[
                index=[
                    "id" -> pd.Categorical[str]
                        "{current_name}@/{sport}/{country}/{name-year}/",
                ],\n
                columns=[
                    "real" -> np.float64
                        "real": gini index for real tournaments),\n
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
                        f"s{i}"   : gini index for i-th simulation,
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
    ) -> Gini:
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

        def _calculate_gini_index_per_id(df: pd.DataFrame) -> pd.DataFrame:
            def _get_upper_bound():
                _df = build_most_imbalanced_tournament(df)
                _rankings = _df.groupby(["id", "team"], observed=True).sum()
                return _rankings.groupby("id", observed=True).apply(gini)

            rankings = df.groupby(["id", "team"], observed=True).sum()

            gini_index = rankings.groupby("id", observed=True).apply(gini)
            normalized_gini = gini_index / _get_upper_bound()
            return np.clip(normalized_gini, 0, 1)

        parameters = get_kwargs_from_points_per_match(
            ppm,
            _calculate_gini_index_per_id,
            num_iteration_simulation,
            id_to_probabilities,
        )
        return cls(**parameters)
