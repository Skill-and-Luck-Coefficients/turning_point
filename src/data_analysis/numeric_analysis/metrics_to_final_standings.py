from typing import Callable

import pandas as pd

from tournament_simulations.data_structures import Matches, PointsPerMatch

MetricFn = Callable[[pd.DataFrame], float]


def apply_metrics_to_final_standings(
    matches: Matches, metric_functions: dict[str, MetricFn]
) -> pd.DataFrame:
    fn = list(metric_functions.values())
    labels = list(metric_functions.keys())

    ppm = PointsPerMatch.from_home_away_winner(matches.home_away_winner())
    functions: pd.DataFrame = ppm.rankings.groupby("id", observed=True).agg(fn)
    return functions.set_axis(labels, axis="columns")
