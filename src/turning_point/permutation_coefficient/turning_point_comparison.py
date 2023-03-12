from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from turning_point.normal_coefficient import TurningPoint

from .permutation_turning_point import PermutationTurningPoint
from .utils import add_second_level_to_column_names


@dataclass
class TurningPointComparison:

    """
    Stores both normal and permutation turning points for all tournaments.

        normal: TurningPoint
        permutation: PermutationTurningPoint

    comparison:
        pd.DataFrame[
            index=[
                id"    -> pd.Categorical[str]
                    "{current_name}@/{sport}/{country}/{name-year}"
            ],\n
            columns=[  # multi level columns
                "turning point"  -> turning point (int),\n
                    "normal"  -> turning point from original schedule,\n
                    "mean"    -> mean turning point,\n
                    "std"     -> standard deviation,\n
                    "f{p}%"   -> percentiles: p in [2.5, 50, 97.5]
                "%turning point" -> normalized turning point (percentage),\n
                    "normal"  -> turning point from original schedule,\n
                    "mean"    -> mean turning point,\n
                    "std"     -> standard deviation,\n
                    "f{p}%"   -> percentiles: p in [2.5, 50, 97.5]
        ]
    """

    normal: TurningPoint
    permutation: PermutationTurningPoint

    @cached_property
    def comparison(self) -> pd.DataFrame:

        """
        Join normal turning point values and statistical measures from
        permutation turning points.

        ----
        Returns:
            pd.DataFrame[
                index=[
                    id"    -> "{current_name}@/{sport}/{country}/{name-year}"
                ],\n
                columns=[  # multi level columns
                    "turning point"  -> turning point (int),\n
                        "normal"  -> turning point from original schedule,\n
                        "mean"    -> mean turning point,\n
                        "std"     -> standard deviation,\n
                        "f{p}%"   -> percentiles: p in [2.5, 50, 97.5]
                    "%turning point" -> normalized turning point (percentage),\n
                        "normal"  -> turning point from original schedule,\n
                        "mean"    -> mean turning point,\n
                        "std"     -> standard deviation,\n
                        "f{p}%"   -> percentiles: p in [2.5, 50, 97.5]
            ]
        """

        extended_normal = add_second_level_to_column_names(self.normal.df, "normal")
        permutation_stats = self.permutation.statistical_measures

        data_type = {"id": "category"}
        joined = (
            pd.concat([extended_normal, permutation_stats], axis=1)
            .reset_index("id")
            .astype(data_type)
            .set_index("id")
        )

        return joined.sort_index().loc(axis="columns")[  # select in the desired order
            ["turning point", "%turning point"],
            ["normal", "mean", "std", "2.5%", "50%", "97.5%"],
        ]
