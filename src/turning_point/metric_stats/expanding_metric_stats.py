from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import pandas as pd

from tournament_simulations.data_structures import Matches
from turning_point.metrics import Metric, Variances

from .calculate_expanding_metric_stats import get_kwargs_expanding_from_matches
from .metric_stats import RESULT_TO_POINTS


@dataclass
class ExpandingMetricStats:

    """
    Contains statistical information about ranking-variances over all simulations.
    In particular, it contains mean and 0.95-quantile.

        df: pd.DataFrame[
            index=[
                "id"    -> "{current_name}@/{sport}/{country}/{name-year}/"\n
                "final date" -> last date considered before simulating\n
            ],\n
            columns=[
                "real" -> metric for real tournament,\n
                "mean" -> mean metric over all simulations,\n
                "quantile"-> quantile metric over all simulations
            ]
        ]
    """

    df: pd.DataFrame

    def __post_init__(self) -> None:
        index_cols = ["id", "final date"]

        index_to_reset = [name for name in index_cols if name in self.df.index.names]
        self.df = self.df.reset_index(index_to_reset)

        data_types = {"id": "category"}
        self.df = self.df.astype(data_types).set_index(index_cols).sort_index()

    @classmethod
    def from_matches(
        cls,
        matches: Matches,
        num_iteration_simulation: tuple[int, int],
        winner_type: Literal["winner", "result"] = "winner",
        winner_to_points: Mapping[str, tuple[float, float]] = RESULT_TO_POINTS,
        id_to_probabilities: pd.Series | None = None,
        quantile: float | None = None,
        metric_type: type[Metric] = Variances,
    ) -> ExpandingMetricStats:
        """
        Create an instance of ExpandingMetricStats from Matches.

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
        """

        params = get_kwargs_expanding_from_matches(
            matches,
            num_iteration_simulation,
            winner_type,
            winner_to_points,
            id_to_probabilities,
            quantile,
            metric_type,
        )
        return cls(**params)
