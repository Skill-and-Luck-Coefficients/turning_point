import warnings

import numpy as np
import pandas as pd
from scipy.stats import linregress, theilslopes


def _get_linear_regression_slope(y: pd.Series) -> float:
    return linregress(np.arange(len(y)), y).slope


def _get_theil_slope(y: pd.Series) -> float:
    return theilslopes(y).slope


def _calculate_tendecy_all_tournaments(turning_point: pd.Series) -> pd.DataFrame:
    regex_tournament_id = r"(.+?@/.+?/.+?/).+"
    tournament_id = turning_point.index.str.extract(regex_tournament_id, expand=False)

    tendency = pd.concat(
        [
            turning_point.groupby(tournament_id).apply(_get_linear_regression_slope),
            turning_point.groupby(tournament_id).apply(_get_theil_slope),
            turning_point.groupby(tournament_id).size(),
        ],
        axis="columns",
        keys=["linregress slope", "theil slope", "#seasons"],
    )
    return tendency


def _is_a_relevant_league(string: str, important_leagues: list[tuple[str]]) -> bool:
    is_important = any(
        all(id_ in string for id_ in league_identifiers)
        for league_identifiers in important_leagues
    )
    return "background-color:red;color:white" if is_important else ""


def get_temporal_slope_tendency(
    sport_to_comparison_df: dict[str, pd.DataFrame],
    column: tuple[str, str] = ("%turning point", "mean"),
) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sport_to_tendency = pd.concat(
            {
                sport: _calculate_tendecy_all_tournaments(comparison_df[column])
                for sport, comparison_df in sport_to_comparison_df.items()
            }
        )

        drop_columns = ["linregress slope", "theil slope"]
        sport_to_tendency = sport_to_tendency.dropna(how="all", subset=drop_columns)
    return sport_to_tendency


def style_temporal_tendency(
    sport_to_tendency: dict[str, pd.DataFrame], important_leagues: list[tuple[str]]
) -> pd.DataFrame:
    col_to_format = {
        "linregress slope": "{:.2%}".format,
        "theil slope": "{:.2%}".format,
    }
    background_gradient_kwargs = {
        "axis": None,
        "vmin": -0.05,
        "vmax": 0.05,
        "cmap": "viridis",
        "subset": ["linregress slope", "theil slope"],
    }
    return (
        sport_to_tendency.style.format(col_to_format)
        .background_gradient(**background_gradient_kwargs)
        .applymap_index(
            lambda s: _is_a_relevant_league(s, important_leagues),
            axis="index",
            level="id",
        )
    )
