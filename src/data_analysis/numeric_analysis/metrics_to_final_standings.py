from typing import Callable

import pandas as pd

from tournament_simulations.data_structures import Matches, PointsPerMatch
from turning_point.metrics.metric import Metric

MetricFn = Callable[[pd.DataFrame], float]


def apply_metrics_to_final_standings(
    matches: Matches, name_to_metric: dict[str, Metric]
) -> pd.DataFrame:
    def _apply_to_final_standings(_name: str, _metric: type[Metric]) -> pd.DataFrame:
        _metric_result = _metric.from_points_per_match(ppm, num_iteration_simulation)
        return _metric_result.real.rename(columns={"real": _name})

    num_iteration_simulation = (0, 0)
    ppm = PointsPerMatch.from_home_away_winner(matches.home_away_winner())
    return pd.concat(
        [
            _apply_to_final_standings(name, metric)
            for name, metric in name_to_metric.items()
        ],
        axis="columns",
    )
