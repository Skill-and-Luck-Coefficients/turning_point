from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tournament_simulations.data_structures import Matches

from ..variances.variances import Variances
from .calculate_variance_stats import get_kwargs_from_matches, get_kwargs_from_variances


@dataclass
class SimulationVarStats:

    """
    Contains statistical information ranking-variances over all simulations.
    In particular, it contains mean value and 0.95-quantile.

        df: pd.DataFrame[
            index=[
                "id"    -> pd.Categorical[str]
                    "{current_name}@/{sport}/{country}/{name-year}/"
            ],\n
            columns=[
                "real var" -> ranking-variance for real tournament,\n
                "mean" -> mean ranking-variance over all simulations,\n
                "0.950-quantile"-> 0.950-quantile ranking-variance over all simulations
            ]
        ]
    """

    df: pd.DataFrame

    def __post_init__(self) -> None:

        index_cols = ["id"]

        index_to_reset = [name for name in index_cols if name in self.df.index.names]
        self.df = self.df.reset_index(index_to_reset)

        data_types = {"id": "category"}
        self.df = self.df.astype(data_types).set_index(index_cols).sort_index()

    @classmethod
    def from_variances(cls, variances: Variances) -> SimulationVarStats:

        """
        Create an instance of SimulationVarStats from variances.

        ----
        Parameters:

            variances: Variances
                Ranking-variances for real and simulated tournaments.
        """
        parameters = get_kwargs_from_variances(variances)
        return cls(**parameters)

    @classmethod
    def from_matches(
        cls, matches: Matches, num_iteration_simulation: tuple[int, int]
    ) -> SimulationVarStats:

        """
        Create an instance of SimulationVarStats from Matches.

        -----
        Parameters:

            matches: Matches
                Tournament matches.

            num_iteration_simulation: tuple[int, int]
                Respectively, number of iterations and number of
                simulations per iteration (batch size).
        """
        parameters = get_kwargs_from_matches(matches, num_iteration_simulation)
        return cls(**parameters)
