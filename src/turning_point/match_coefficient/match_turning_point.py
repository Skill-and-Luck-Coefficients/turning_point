from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tournament_simulations.data_structures import Matches
from turning_point.normal_coefficient import TurningPoint

from .create_match_turning_point import get_kwargs_from_matches_turning_point


@dataclass
class MatchTurningPoint:

    """
    Stores number of matches played until turning point date.

    matches_tp:
        pd.DataFrame[
            index=[
                id"    -> "{current_name}@/{sport}/{country}/{name-year}"
            ],\n
            columns=[
                "matches turning point": int
                    Number of matches until turning point date
                "%matches turning point": float (percentage)
                    Normalized matches turning point
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
    def from_matches_and_turning_point(
        cls, matches: Matches, turning_point: TurningPoint
    ) -> MatchTurningPoint:

        """
        Creates an instance of MatchTurningPoint.

        -----
        Parameters:

            matches: Matches
                Tournament matches.

            turning_point: TurningPoint:
                Normal turning point.
        """

        parameters = get_kwargs_from_matches_turning_point(matches, turning_point)
        return cls(**parameters)
