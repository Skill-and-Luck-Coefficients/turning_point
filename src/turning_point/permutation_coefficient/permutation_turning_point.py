import functools
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

    @functools.cached_property
    def statistical_measures(self) -> pd.DataFrame:

        """
        Calculate mean, standard deviation and percentiles (2.5%, 25%, 50%, 75%, 97.5%)
        over all permutations.

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
                        "f{p}%" -> percentiles: p in [2.5, 25, 50, 75, 97.5]
                    "%turning point"  -> turning point\n
                        "mean"  -> mean turning point,\n
                        "std"   -> standard deviation,\n
                        "f{p}%" -> percentiles: p in [2.5, 25, 50, 75, 97.5]
            ]
        """
        # rename mapper function
        def _rename(id_: str) -> str:
            *original_id, _ = id_.split("@")
            return "@".join(original_id)

        percentiles = [2.5, 25, 50, 75, 97.5]
        desired_measures = ["mean", "std"] + [f"{p}%" for p in percentiles]

        return (
            self.df.rename(index=_rename)
            .groupby("id", observed=True)
            .describe(percentiles=[p / 100 for p in percentiles])
            .loc(axis=1)[:, desired_measures]
        )
