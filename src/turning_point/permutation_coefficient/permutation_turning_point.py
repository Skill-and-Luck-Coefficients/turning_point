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
                        "{current_name}@/{sport}/{country}/{name-year}@{permutation_id}"
                ],\n
                columns=[
                    "turning point"  -> turning point (int),\n
                    "%turning point" -> normalized turning point (percentage),\n
            ]
    """

    def unstack_permutation_id(self) -> pd.DataFrame:
        """
        Returns a dataframe in which each permutation_id has its own column for each id.

        ----
        Returns:
            pd.DataFrame[
                index=[
                    id"    -> "{current_name}@/{sport}/{country}/{name-year}"
                ],\n
                columns=[  # multi level columns
                    "turning point"  -> turning point\n
                        "{1st permutation_id}"  -> turning point,\n
                        "{2nd permutation_id}"  -> turning point,\n
                        ...\n
                        "{n-th permutation_id}" -> turning point,\n
                    "%turning point"  -> turning point\n
                        "{1st permutation_id}"  -> turning point,\n
                        "{2nd permutation_id}"  -> turning point,\n
                        ...\n
                        "{n-th permutation_id}" -> turning point,\n
            ]

        """
        # (.+?@.+?)@(.+)
        #   tournament id: (greedy) first group
        #   permutation id: second group
        new_index_df = self.df.index.str.extract("(.+)@(.+)")
        new_index = pd.MultiIndex.from_frame(new_index_df, names=["id", "type"])

        return self.df.set_index(new_index).unstack("type")  # type: ignore

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
