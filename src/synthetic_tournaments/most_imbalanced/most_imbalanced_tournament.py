import numpy as np
import pandas as pd

import tournament_simulations.data_structures as ds


def build_most_imbalanced_tournament_one_id(ppm_df: pd.DataFrame) -> pd.DataFrame:
    def _get_indexes_team(_team):
        """
        Get indexes (integers) where df['team'] == team.
        """
        return np.nonzero((ppm_df["team"] == _team).to_numpy())[0]

    def _set_match_results(_indexes):
        """
        3: in the indexes for `team`
        0: in the indexes for the adversary
            team_index + 1 for even team_index
            team_index - 1 for odd team_index

        """
        points[_indexes] = 3

        _even_indexes = _indexes[_indexes % 2 == 0]
        points[_even_indexes + 1] = 0

        _odd_indexes = _indexes[_indexes % 2 != 0]
        points[_odd_indexes - 1] = 0

    points = np.zeros_like(ppm_df["team"])
    teams = ppm_df.groupby("team", observed=True).size().sort_values()

    for team in teams.index:
        indexes = _get_indexes_team(team)
        _set_match_results(indexes)

    data = {"team": ppm_df["team"], "point": points}
    return pd.DataFrame(data, index=ppm_df.index)


def build_most_imbalanced_tournament(
    ppm: pd.DataFrame | ds.PointsPerMatch,
) -> pd.DataFrame:
    """
    For a given tournament (id):
        Team that played the most amount of matches: win all matches
        Team that played the second most amount of matches: win all matches (except for the previous team)
        ...
    """
    if isinstance(ppm, ds.PointsPerMatch):
        df = ppm.df

    build_one_id_fn = build_most_imbalanced_tournament_one_id
    return df.groupby("id", observed=True).apply(build_one_id_fn)
