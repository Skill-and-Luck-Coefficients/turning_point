import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

import tournament_simulations.schedules.round_robin as ts_rr
from tournament_simulations.data_structures import Matches
from tournament_simulations.schedules import convert_list_of_rounds_to_dataframe
from turning_point.metric_stats import ExpandingMetricStats
from turning_point.normal_coefficient import TurningPoint


@dataclass
class ContainDF:
    df: pd.DataFrame


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _swap_column_levels(df: pd.DataFrame) -> pd.DataFrame:
    return df.swaplevel(0, 1, "columns").sort_index(axis="columns")


def _get_winner(
    matches: pd.DataFrame,
    strengths: list[float],
) -> pd.Series:
    strengths_ = dict(enumerate(strengths))
    home_skill_per_match = matches["home"].map(strengths_)
    away_skill_per_match = matches["away"].map(strengths_)

    prob_home_win = home_skill_per_match / (home_skill_per_match + away_skill_per_match)
    uniform_values = np.random.random(size=len(matches))
    return (uniform_values <= prob_home_win).map({True: "h", False: "a"})


def _create_schedule(
    strengths: list[float], label: str, number_of_tournaments: int
) -> pd.DataFrame:
    drr = ts_rr.DoubleRoundRobin.from_num_teams(len(strengths))
    schedule = drr.get_full_schedule(number_of_tournaments)
    matches_df = convert_list_of_rounds_to_dataframe(schedule, label)
    matches_df["winner"] = _get_winner(matches_df, strengths)
    return matches_df


def run_bradley_terry_simulations(
    strengths: dict[str, list[int]],
    num_simulations: int = 10,
    seed: int = 0,
    number_of_tournaments: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        tuple[
            pd.DataFrame, # metric stats quantile 0.05
            pd.DataFrame, # metric stats quantile 0.95
            pd.DataFrame, # turning points
        ]
    """

    various_tps = {}
    various_stats = {}
    various_stats_lower = {}

    seeds = range(seed, seed + num_simulations)
    num_simulations_ = range(num_simulations)

    for seed_, num_simulation in zip(seeds, num_simulations_):
        _set_seed(seed_)
        matches_df_list = [
            _create_schedule(strength, f"bradley_terry/{label}", number_of_tournaments)
            for label, strength in strengths.items()
        ]
        matches = Matches(pd.concat(matches_df_list))

        _set_seed(seed_)
        parameters = {"num_iteration_simulation": (10, 100), "quantile": 0.05}
        expanding_var = ExpandingMetricStats.from_matches(matches, **parameters)
        various_stats_lower[num_simulation] = expanding_var.df

        _set_seed(seed_)
        parameters = {"num_iteration_simulation": (10, 100), "quantile": 0.95}
        expanding_var = ExpandingMetricStats.from_matches(matches, **parameters)
        various_stats[num_simulation] = expanding_var.df

        turning_point = TurningPoint.from_expanding_var_stats(expanding_var)
        various_tps[num_simulation] = turning_point.df

    stats = _swap_column_levels(pd.concat(various_stats, axis="columns"))
    stats_lower = _swap_column_levels(pd.concat(various_stats_lower, axis="columns"))
    tps = _swap_column_levels(pd.concat(various_tps, axis="columns"))
    return stats_lower, stats, tps


def filter_stats_one_simulation(
    df: pd.DataFrame, simulation_number: int
) -> dict[str, ContainDF]:
    def _remove_bradley_terry_prefix(string: str) -> str:
        return string.split("/")[1]

    column_slice = pd.IndexSlice[:, simulation_number]
    tournament_names = df.index.get_level_values("id").unique()
    return {
        _remove_bradley_terry_prefix(name): ContainDF(
            df.loc[[name], column_slice].droplevel(1, axis="columns")
        )
        for name in tournament_names
    }