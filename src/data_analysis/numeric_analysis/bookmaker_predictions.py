from typing import Sequence

import pandas as pd

from tournament_simulations.data_structures import Matches

ALL_ODDS_COLUMNS = ["odds home", "odds away", "odds tie"]

ODDS_COLUMNS_TO_RESULT = {
    "odds home": "h",
    "odds away": "a",
    "odds tie": "d",
}


def _get_ratio_with_odds(
    matches: pd.DataFrame, odds_columns: Sequence[str] | pd.Index
) -> pd.Series:
    has_odds = matches[odds_columns].notna().all(axis="columns")

    id_to_counter = has_odds.groupby("id", observed=True).sum()
    id_to_ratio = id_to_counter / has_odds.groupby("id", observed=True).size()

    return id_to_ratio.rename("%odds present")


def _get_percent_bookmaker_correct(
    matches: pd.DataFrame, odds_columns: Sequence[str] | pd.Index
) -> pd.Series:
    favorite_to_win = matches[odds_columns].idxmin(axis="columns")
    winner_betexplorer = favorite_to_win.map(ODDS_COLUMNS_TO_RESULT)

    betexplorer_right = winner_betexplorer == matches["winner"]
    betexplorer_right[winner_betexplorer.isna()] = pd.NA

    return betexplorer_right.groupby("id", observed=True).mean().rename("%bookmarker")


def get_sport_to_bookmakers_comparison(
    sport_to_matches: dict[str, Matches],
    sport_to_tp_comparison: dict[str, pd.DataFrame],
    tp_column: tuple[str, str] = ("%turning point", "mean"),
    min_present: float = 0.85,
) -> dict[str, pd.DataFrame]:
    sport_to_bookmarker_comparison: dict[str, pd.DataFrame] = {}

    for sport in sport_to_matches.keys():
        matches = sport_to_matches[sport].df
        turning_point = sport_to_tp_comparison[sport][tp_column]

        odds_cols = matches.columns.intersection(ALL_ODDS_COLUMNS)

        id_to_odd_percent = _get_ratio_with_odds(matches, odds_cols)
        bookmarker_right = _get_percent_bookmaker_correct(matches, odds_cols)

        to_concat = [bookmarker_right, turning_point, id_to_odd_percent]
        concated_df = pd.concat(to_concat, axis="columns")

        more_than_min = min_present <= concated_df["%odds present"]
        sport_to_bookmarker_comparison[sport] = concated_df[more_than_min]

    return sport_to_bookmarker_comparison
