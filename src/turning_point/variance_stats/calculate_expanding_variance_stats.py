from typing import Callable, Concatenate, Literal, ParamSpec

import numpy as np
import pandas as pd

from logs import log, turning_logger
from tournament_simulations.data_structures import Matches

from .get_matches_in_window import select_matches_inside_window
from .variance_stats import SimulationVarStats

KwargsEW = dict[Literal["df"], pd.DataFrame]
P = ParamSpec("P")


def _expanding_template(
    matches: Matches,
    expanding_func: Callable[Concatenate[Matches, P], pd.DataFrame],
    *args: P.args,
    **kwargs: P.kwargs
) -> pd.DataFrame:

    """
    Creating this template makes testing easier, since 'expanding_func'
    can be a simple function.

    'expanding_func' is the function to be applied in each expading window.
    """

    all_results: list[pd.DataFrame] = []

    last_date = matches.df.index.get_level_values("date number").max()

    for date in range(last_date + 1):

        matches_window = select_matches_inside_window(
            matches, first_date=0, last_date=date
        )

        df = expanding_func(matches_window, *args, **kwargs)
        df["final date"] = np.int16(date)  # save corresponding date

        all_results.append(df)

    return pd.concat(all_results)


@log(turning_logger.debug)
def get_kwargs_expanding_from_matches(
    matches: Matches, num_iteration_simulation: tuple[int, int]
) -> KwargsEW:

    """
    Iterated dynamic skill coefficient and removed teams for all possible
    expanding windows:
        [0], [0, 1], [0, 1, 2], ..., [0, 1, ..., T - 1, T]
    where T is the last date of a given tournament.

    Remark: Since we simulate new tournaments every date,
             pandas' .expanding() method cannot be used.

    -----
    Parameters:

        matches: MatchesSlice
            Matches each team played in the tournament in the respective date
            number interval.

        num_iteration_simulation: tuple[int, int]
            Respectively, number of iterations and number of
            simulations per iteration (batch size).

    -----
    Returns:
        Kwargs parameters required to create and instance of ExpandingCoefAndTeams
            "df": IteratedSkillCoefAndTeams dataframe for all windows.
    """

    def _simulation_var_stats(*args, **kwargs) -> pd.DataFrame:
        # _expanding_template expects a function that returns a dataframe
        return SimulationVarStats.from_matches(*args, **kwargs).df

    # easier to test with the template since all randomness in _simulation_var_stats
    expading_df = _expanding_template(
        matches=matches,
        expanding_func=_simulation_var_stats,
        # other parameters for expanding_func
        num_iteration_simulation=num_iteration_simulation,
    )

    return {"df": expading_df}
