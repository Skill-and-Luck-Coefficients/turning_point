from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tournament_simulations.data_structures import Matches

from .calculate_expanding_variance_stats import get_kwargs_expanding_from_matches


@dataclass
class ExpandingVarStats:

    """
    Contains statistical information about ranking-variances over all simulations.
    In particular, it contains mean and 0.95-quantile.

        df: pd.DataFrame[
            index=[
                "id"    -> "{current_name}@/{sport}/{country}/{name-year}/"\n
                "final date" -> last date considered before simulating\n
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
        index_cols = ["id", "final date"]

        index_to_reset = [name for name in index_cols if name in self.df.index.names]
        self.df = self.df.reset_index(index_to_reset)

        data_types = {"id": "category"}
        self.df = self.df.astype(data_types).set_index(index_cols).sort_index()

    @classmethod
    def from_matches(
        cls,
        matches: Matches,
        num_iteration_simulation: tuple[int, int],
        id_to_probabilities: pd.Series | None = None,
    ) -> ExpandingVarStats:
        """
        Create an instance of ExpandingSimulVarStats from Matches.

        -----
        Parameters:

            matches: MatchesSlice
                Matches each team played in the tournament in the respective date
                number interval.

            num_iteration_simulation: tuple[int, int]
                Respectively, number of iterations and number of
                simulations per iteration (batch size).

            id_to_probabilities: pd.Series | None = None
                Series mapping each tournament to its estimated probabilities.

                Probabilities: (prob home win, prob draw, prob away win).

                If None, they will be estimated directly from 'matches'.
        """

        params = get_kwargs_expanding_from_matches(
            matches, num_iteration_simulation, id_to_probabilities
        )
        return cls(**params)
