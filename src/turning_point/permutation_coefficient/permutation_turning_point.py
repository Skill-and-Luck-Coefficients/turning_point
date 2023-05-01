from dataclasses import dataclass

import pandas as pd

from turning_point.normal_coefficient import TurningPoint


@dataclass
class PermutationTurningPoint(TurningPoint):

    """
    Turning point for permutations of tournaments.

    Inherits from TurningPoint.

        df:
            pd.DataFrame[
                index=[
                    id" -> pd.Categorical[str]
                        "{current_name}@/{sport}/{country}/{name-year}@{num_permutation}"
                ],\n
                columns=[
                    "turning point"  -> turning point (int),\n
                    "%turning point" -> normalized turning point (percentage),\n
            ]
    """

    def statistical_measures(self, percentiles: list[float]) -> pd.DataFrame:

        """
        Calculate mean, standard deviation and percentiles.
        over all permutations.

        ---
        Parameters:
            percentiles: list[float]
                Percentile values to calculate.

                Percentiles should fall between 0 and 100.

        ---
        Returns:
            pd.DataFrame[
                index=[
                    id"    -> "{current_name}@/{sport}/{country}/{name-year}"
                ],\n
                columns=[  # multi level columns
                    "turning point"  -> turning point\n
                        "mean"  -> mean turning point,\n
                        "std"   -> standard deviation,\n
                        "f{p}%" -> percentiles: p in 'percentiles'
                    "%turning point"  -> turning point\n
                        "mean"  -> mean turning point,\n
                        "std"   -> standard deviation,\n
                        "f{p}%" -> percentiles: p in 'percentiles'
            ]
        """
        # rename mapper function
        def _rename(id_: str) -> str:
            *original_id, _ = id_.split("@")
            return "@".join(original_id)

        desired_measures = ["mean", "std"] + [f"{p}%" for p in sorted(percentiles)]

        return (
            self.df.rename(index=_rename)
            .groupby("id", observed=True)
            .describe(percentiles=[p / 100 for p in percentiles])
            .loc(axis=1)[:, desired_measures]
        )
