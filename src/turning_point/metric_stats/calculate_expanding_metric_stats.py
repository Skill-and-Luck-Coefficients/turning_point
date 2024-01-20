from typing import Callable, Concatenate, Literal, Mapping, ParamSpec

import numpy as np
import pandas as pd

from logs import log, log_iterations, turning_logger
from tournament_simulations.data_structures import Matches
from turning_point.metrics import Metric, Variances

from .get_matches_in_window import select_matches_inside_window
from .metric_stats import RESULT_TO_POINTS, SimulationMetricStats

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

    dates = range(last_date + 1)
    for date in log_iterations(dates, turning_logger.info, every_n=10):
        matches_window = select_matches_inside_window(
            matches, first_date=0, last_date=date
        )

        df = expanding_func(matches_window, *args, **kwargs)
        df["final date"] = np.int16(date)  # save corresponding date

        all_results.append(df)

    return pd.concat(all_results)


@log(turning_logger.debug)
def get_kwargs_expanding_from_matches(
    matches: Matches,
    num_iteration_simulation: tuple[int, int],
    winner_type: Literal["winner", "result"] = "winner",
    winner_to_points: Mapping[str, tuple[float, float]] = RESULT_TO_POINTS,
    id_to_probabilities: pd.Series | None = None,
    quantile: float | None = None,
    metric_type: type[Metric] = Variances,
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

        winner_type: Literal["winner", "result"] = "winner"
            What should points be based on.
                match: winner of the match
                    home: "h"
                    draw: "d"
                    away: "a"
                result: result of match: f{score home team}-{score away team}"

        winner_to_points: Mapping[str, tuple[float, float]]
            Mapping winner/result to how many points each team should gain.

            First tuple value is for home-team and the second one for away-team.

            Default: {"h": (3, 0), "d": (1, 1), "a": (0, 3)}
                h: home team (3 points); away team (0 points)
                d: home team (1 point); away team (1 point)
                a: home team (0 points); away team (3 points)

        id_to_probabilities: pd.Series | None = None
            Series mapping each tournament to its estimated probabilities.

            Probabilities:  Mapping[tuple[float, float]: float]
                Maps each pair (tuple) to its probability (float).

                Pair: ranking points gained respectively by home-team and away-team.

            If None, they will be estimated directly from 'matches'.

        quantile: float | None = None
            Desired quantile value.

            If None, defaults to 0.95.

        metric_type: type[Metric] = Variances
            Which metric should be used.

    -----
    Returns:
        Kwargs parameters required to create and instance of ExpandingCoefAndTeams
            "df": IteratedSkillCoefAndTeams dataframe for all windows.
    """

    def _simulation_var_stats(*args, **kwargs) -> pd.DataFrame:
        # _expanding_template expects a function that returns a dataframe
        return SimulationMetricStats.from_matches(*args, **kwargs).df

    # easier to test with the template since all randomness in _simulation_var_stats
    expading_df = _expanding_template(
        matches=matches,
        expanding_func=_simulation_var_stats,
        # other parameters for expanding_func
        num_iteration_simulation=num_iteration_simulation,
        winner_type=winner_type,
        winner_to_points=winner_to_points,
        id_to_probabilities=id_to_probabilities,
        quantile=quantile,
        metric_type=metric_type,
    )

    return {"df": expading_df}
