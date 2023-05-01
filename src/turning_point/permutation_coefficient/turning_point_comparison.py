from dataclasses import dataclass
from itertools import product

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
                    "f{p}%"   -> desired percentiles
                "%turning point" -> normalized turning point (percentage),\n
                    "normal"  -> turning point from original schedule,\n
                    "mean"    -> mean turning point,\n
                    "std"     -> standard deviation,\n
                    "f{p}%"   -> desired percentiles
        ]
    """

    normal: TurningPoint
    permutation: PermutationTurningPoint

    def comparison(self, percentiles: list[float]) -> pd.DataFrame:

        """
        Join normal turning point values and statistical measures from
        permutation turning points.

        Parameters:
            percentiles: list[float]
                Percentile values to calculate.

                Percentiles should fall between 0 and 100.

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
                        "f{p}%"   -> percentiles: p in 'percentiles'
                    "%turning point" -> normalized turning point (percentage),\n
                        "normal"  -> turning point from original schedule,\n
                        "mean"    -> mean turning point,\n
                        "std"     -> standard deviation,\n
                        "f{p}%"   -> percentiles: p in 'percentiles'
            ]
        """

        extended_normal = add_second_level_to_column_names(self.normal.df, "normal")
        permutation_stats = self.permutation.statistical_measures(percentiles)

        data_type = {"id": "category"}
        joined = (
            pd.concat([extended_normal, permutation_stats], axis=1)
            .reset_index("id")
            .astype(data_type)
            .set_index("id")
        )

        col_level_one_order = ["turning point", "%turning point"]

        stats_cols: pd.DataFrame = permutation_stats["turning point"]
        col_level_two_order = ["normal"] + stats_cols.columns.to_list()

        col_order = list(product(col_level_one_order, col_level_two_order))
        return joined.sort_index()[col_order]
