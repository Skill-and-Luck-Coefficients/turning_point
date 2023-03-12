from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from turning_point.variance_stats import ExpandingVarStats

from .calculate_turning_point import get_kwargs_from_expanding_variances_stats


@dataclass
class TurningPoint:

    """
    Stores turning point calculated for all tournaments.

        df:
            pd.DataFrame[
                index=[
                    id"    -> pd.Categorical[str]
                        "{current_name}@/{sport}/{country}/{name-year}"
                ],\n
                columns=[
                    "turning point"  -> turning point (int),\n
                    "%turning point" -> normalized turning point (percentage),\n
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
    def from_expanding_var_stats(
        cls, expanding_stats: ExpandingVarStats
    ) -> TurningPoint:

        """
        Creates an instance of TurningPoint from expanding variances stats.

        -----
        Parameters:

            expanding_var: ExpandingVarStats:

                1) Real variances for all tournaments.
                2) Statistical data (mean and 0.95-quantile) over all simulations.

                Each date number window starting at zero has it own value.
        """

        parameters = get_kwargs_from_expanding_variances_stats(expanding_stats)
        return cls(**parameters)
