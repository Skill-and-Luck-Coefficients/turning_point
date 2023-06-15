from typing import Literal, Sequence

import numpy as np
import pandas as pd

from logs import log, turning_logger
from turning_point.variance_stats import ExpandingVarStats

KwargsTP = dict[Literal["df"], pd.DataFrame]


def _find_turning_point_one_id(sequence: Sequence[bool]) -> float:
    """
    Turning point is the index for a sequence of True until the end.
    """

    if len(sequence) == 0:
        return np.nan

    # if the last position is False, there can't be such a sequence
    if not sequence[-1]:
        return np.inf

    # finding a sequence of true in the end is related to
    # finding the first occurrence of False in the reversed iterator
    for index, boolean in enumerate(reversed(sequence)):
        if not boolean:
            return len(sequence) - index
    return 0


def _find_turning_point_percent_one_id(sequence: Sequence[bool]) -> float:
    if len(sequence) == 0:
        return np.nan

    return (_find_turning_point_one_id(sequence) + 1) / len(sequence)


@log(turning_logger.debug)
def get_kwargs_from_expanding_variances_stats(
    expanding_var: ExpandingVarStats,
) -> KwargsTP:
    """
    Calculate turning point.

    -----
    Parameters:

        expanding_var: ExpandingVarStats:

            1) Real variances for all tournaments.
            2) Statistical data (mean and 0.95-quantile) over all simulations.

            Each date number window starting at zero has it own value.

    ----
    Returns:
        Returns kwargs required to create an instance of TurningPoint:
            "df": Turning point values for all tournaments.
    """

    real_var = expanding_var.df["real var"]
    simul_quantile = expanding_var.df["0.950-quantile"]

    real_greater_quantile = real_var > simul_quantile

    turning_point = (  # agg groups together all final dates booleans
        real_greater_quantile.groupby("id", observed=True)
        .agg([_find_turning_point_one_id, _find_turning_point_percent_one_id])
        .set_axis(["turning point", "%turning point"], axis="columns")
        .sort_index()
    )

    return {"df": turning_point}
