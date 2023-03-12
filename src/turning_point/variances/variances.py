from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tournament_simulations.data_structures import PointsPerMatch

from .calculate_variances import get_kwargs_from_points_per_match


@dataclass
class Variances:
    """
    Tournament-rankings variances.

        real:
            pd.DataFrame[
                index=[
                    "id" -> pd.Categorical[str]
                        "{current_name}@/{sport}/{country}/{name-year}/",
                ],\n
                columns=[
                    "real var" -> np.float64
                        "real var": ranking variance for real tournaments),\n
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
                        f"s{i}"   : ranking variance for i-th simulation,
                ]
            ]
    """

    real: pd.DataFrame
    simulated: pd.DataFrame

    def __post_init__(self) -> None:

        index_cols = ["id"]

        to_reset = [name for name in index_cols if name in self.real.index.names]
        self.real = self.real.reset_index(to_reset)

        to_reset = [name for name in index_cols if name in self.simulated.index.names]
        self.simulated = self.simulated.reset_index(to_reset)

        data_types = {"id": "category"}
        self.real = self.real.astype(data_types).set_index(index_cols).sort_index()
        self.simulated = (
            self.simulated.astype(data_types).set_index(index_cols).sort_index()
        )

    @classmethod
    def from_points_per_match(
        cls,
        ppm: PointsPerMatch,
        num_iteration_simulation: tuple[int, int],
    ) -> Variances:

        """
        Creates an instance from PointsPerMatch.

        -----
        Parameters:

            points_per_match: PointsPerMatch
                Points each team gained in each match they played.

            num_iteration_simulation: tuple[int, int]
                Respectively, number of iterations and number
                of simulations per iteration (batch size).
        """

        parameters = get_kwargs_from_points_per_match(ppm, num_iteration_simulation)
        return cls(**parameters)
