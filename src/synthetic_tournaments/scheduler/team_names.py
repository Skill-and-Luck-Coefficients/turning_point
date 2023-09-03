import pandas as pd

from tournament_simulations.data_structures import Matches, PointsPerMatch

ID_EXCEPT_YEAR = r"(.+?@/.+?/.+?/).+"


def _aggregate_teams_names_per_id(df: pd.DataFrame) -> pd.Series:
    def _get_teams_from_index(df: pd.DataFrame) -> list[str]:
        return df.index.get_level_values("team").to_list()

    return df.groupby("id", observed=True).apply(_get_teams_from_index)


def from_current_rankings(matches: Matches) -> pd.Series:
    """
    For each id, get teams sorted by its final ranking.
    """
    rankings = PointsPerMatch.from_home_away_winner(matches.home_away_winner).rankings

    by = ["id", "points", "team"]
    sorted_rankings = rankings.sort_values(by, ascending=[True, False, True])
    return _aggregate_teams_names_per_id(sorted_rankings)
