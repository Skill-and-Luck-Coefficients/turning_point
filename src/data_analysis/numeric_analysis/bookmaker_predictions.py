from typing import Sequence

import pandas as pd

from tournament_simulations.data_structures import Matches

ALL_ODDS_COLUMNS = ["odds home", "odds away", "odds tie"]

ODDS_COLUMNS_TO_RESULT = {
    "odds home": "h",
    "odds away": "a",
    "odds tie": "d",
}


def _select_tournament_with_all_odds(
    matches: pd.DataFrame, odds_columns: Sequence[str] | pd.Index
) -> pd.DataFrame:
    has_odds = matches[odds_columns].notna().all(axis="columns")
    tournament_has_all_odds = has_odds.groupby("id", observed=True).all()

    only_tourney_all_odds = tournament_has_all_odds[tournament_has_all_odds]
    return matches.loc[only_tourney_all_odds.index].copy()


def _get_percent_bookmaker_correct(
    matches_all_odds: pd.DataFrame, odds_columns: Sequence[str] | pd.Index
) -> pd.Series:
    favorite_to_win = matches_all_odds[odds_columns].idxmin(axis="columns")
    winner_betexplorer = favorite_to_win.map(ODDS_COLUMNS_TO_RESULT)

    betexplorer_right = winner_betexplorer == matches_all_odds["winner"]
    return betexplorer_right.groupby("id", observed=True).mean().rename("%bookmarker")


def get_sport_to_bookmakers_comparison(
    sport_to_matches: dict[str, Matches],
    sport_to_tp_comparison: dict[str, pd.DataFrame],
    tp_column: tuple[str, str] = ("%turning point", "mean"),
) -> dict[str, pd.DataFrame]:
    sport_to_bookmarker_comparison: dict[str, pd.DataFrame] = {}

    for sport in sport_to_matches.keys():
        matches = sport_to_matches[sport].df
        turning_point = sport_to_tp_comparison[sport][tp_column]

        odds_cols = matches.columns.intersection(ALL_ODDS_COLUMNS)
        matches_all_odds = _select_tournament_with_all_odds(matches, odds_cols)

        bookmarker_right = _get_percent_bookmaker_correct(matches_all_odds, odds_cols)
        turning_point_all_odds = turning_point.loc[bookmarker_right.index]

        to_concat = [bookmarker_right, turning_point_all_odds]
        sport_to_bookmarker_comparison[sport] = pd.concat(to_concat, axis="columns")

    return sport_to_bookmarker_comparison
