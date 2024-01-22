from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tournament_simulations.data_structures import PointsPerMatch

from .calculate_metric import get_kwargs_from_points_per_match
from .metric import Metric


def top_x_percent_concentration_ratio(
    standings: pd.Series | pd.DataFrame, x: float = 0.25, sort: bool = False
) -> float | pd.Series:
    """
    standings:
        pd.DataFrame: Returns pd.Series for each column
        pd.Series: Returns float value

    sort: bool = False
        True: must sort each ranking
        False: rankings are already sorted
    """
    normalized_standings = standings / standings.sum()
    top_x_percent_size = int(len(normalized_standings) * x)

    if isinstance(standings, pd.Series):
        return normalized_standings.nlargest(top_x_percent_size).sum()

    return pd.Series(
        [
            normalized_standings[col].nlargest(top_x_percent_size).sum()
            for col in normalized_standings
        ],
        index=normalized_standings.columns,
    )


def normalized_top_x_percent_concentration_ratio(
    standings: pd.Series | pd.DataFrame, x: float = 0.25
) -> float | pd.Series:
    """
    standings:
        pd.DataFrame: Returns pd.Series for each column
        pd.Series: Returns float value
    """
    top_x_percent_size = int(len(standings) * x)
    adjusted_x = top_x_percent_size / len(standings)

    top_X_percent = top_x_percent_concentration_ratio(standings, x)

    if adjusted_x > 0:
        return top_X_percent / adjusted_x

    return top_X_percent


def fast_normalized_top_x_percent_concentration_ratio(
    standings: pd.Series | pd.DataFrame, x: float = 0.25
) -> float | pd.Series:
    """
    This functions uses .quantile to determine which tournaments are in the top X%.
    So it is not as accurate.

    standings:
        pd.DataFrame: Returns pd.Series for each column
        pd.Series: Returns float value
    """
    normalized_standings = standings / standings.sum()

    x_quantile = normalized_standings.quantile(1 - x)
    top_index = normalized_standings > x_quantile

    normalization = top_index.sum() / len(top_index)
    return normalized_standings[top_index].sum() / normalization


def _calculate_normalized_top_x_cr_per_id(df: pd.DataFrame) -> pd.DataFrame:
    rankings = df.groupby(["id", "team"], observed=True).sum()
    top_x_cr_fn = fast_normalized_top_x_percent_concentration_ratio
    return rankings.groupby("id", observed=True).apply(top_x_cr_fn)


@dataclass
class ConcentrationRatio(Metric):
    """
    Top X% concentration ratio.

        real:
            pd.DataFrame[
                index=[
                    "id" -> pd.Categorical[str]
                        "{current_name}@/{sport}/{country}/{name-year}/",
                ],\n
                columns=[
                    "real" -> np.float64
                        "real": top concentration ratio for real tournaments),\n
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
                        f"s{i}"   : top concentration ratio for i-th simulation,
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
    ) -> ConcentrationRatio:
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
            _calculate_normalized_top_x_cr_per_id,
            num_iteration_simulation,
            id_to_probabilities,
        )
        return cls(**parameters)
