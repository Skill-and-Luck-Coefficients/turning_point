import pandas as pd

from tournament_simulations.data_structures import Matches


def from_matches(matches: Matches) -> pd.Series:
    """
    Get the number of times each (home, away) match happened.
    """
    match_count = matches.home_vs_away_count_per_id
    return match_count.groupby("id", observed=True).max()
