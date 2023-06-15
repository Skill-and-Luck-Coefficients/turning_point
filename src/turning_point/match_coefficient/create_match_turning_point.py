from typing import Literal

import numpy as np
import pandas as pd

from logs import log, turning_logger
from tournament_simulations.data_structures import Matches
from turning_point.normal_coefficient import TurningPoint

KwargsMTP = dict[Literal["df"], pd.DataFrame]


def _count_number_and_percentage_of_matches_in_interval(
    index_and_turning_point: tuple[str, float], matches: Matches
) -> tuple[float, float]:
    tourney_id, turning_point = index_and_turning_point

    if np.isinf(turning_point) or np.isnan(turning_point):
        return turning_point, turning_point  # return np.inf or np.nan

    # turning point is the the date after changing
    num_matches = len(matches.df.loc(axis=0)[tourney_id, : int(turning_point) - 1])

    return num_matches, num_matches / len(matches.df.loc[tourney_id])


@log(turning_logger.debug)
def get_kwargs_from_matches_turning_point(
    matches: Matches, turning_point: TurningPoint
) -> KwargsMTP:
    """
    Calculate match turning point.

    -----
    Parameters:

        matches: Matches
            Tournament matches.

        turning_point: TurningPoint:
            Normal turning point.

    ----
    Returns:
        Returns kwargs required to create an instance of MatchTurningPoint:
            "df": Match turning point values for all tournaments.
    """

    tp_col = turning_point.df["turning point"]
    id_to_tuple_id__turning_point = pd.Series(index=tp_col.index, data=tp_col.items())

    # returns series of tuples -> (number of matches, percentage of matches)
    num_match_and_percent_tp = id_to_tuple_id__turning_point.apply(
        _count_number_and_percentage_of_matches_in_interval,
        matches=matches,
    )

    percent_match_tp = pd.DataFrame(
        num_match_and_percent_tp.to_list(),
        index=num_match_and_percent_tp.index,
        columns=["match turning point", "%match turning point"],
    )

    return {"df": percent_match_tp}
