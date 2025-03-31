from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass

import pandas as pd

from tournament_simulations.data_structures import PointsPerMatch


@dataclass
class Metric(metaclass=ABCMeta):
    """
    Metric interface.

    real:
            pd.DataFrame[
                index=[
                    "id" -> pd.Categorical[str]
                        "{current_name}@/{sport}/{country}/{name-year}/",
                ],\n
                columns=[
                    "real" -> np.float64
                        "real": metric for real tournament,\n
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
                        f"s{i}"   : metric for i-th simulation,
                ]
            ]
    """

    real: pd.DataFrame
    simulated: pd.DataFrame | None

    def __post_init__(self) -> None:
        def _set_correct_types(_df: pd.DataFrame) -> pd.DataFrame:
            _to_reset = [name for name in index_cols if name in _df.index.names]
            _df = _df.reset_index(_to_reset)
            return _df.astype(dtype).set_index(index_cols).sort_index()

        index_cols = ["id"]
        dtype = {"id": "category"}

        self.real = _set_correct_types(self.real)

        if self.simulated is not None:
            self.simulated = _set_correct_types(self.simulated)

    @classmethod
    def from_points_per_match(
        cls,
        ppm: PointsPerMatch,
        num_iteration_simulation: tuple[int, int],
        id_to_probabilities: pd.Series | None = None,
    ) -> Metric:
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
        ...
