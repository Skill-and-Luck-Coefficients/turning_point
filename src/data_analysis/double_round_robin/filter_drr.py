from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import pandas as pd

from tournament_simulations.data_structures import Matches

T = TypeVar("T")


def _get_matches_one_id(teams: list[T]) -> list[tuple[T, T]]:
    return [(team1, team2) for team1 in teams for team2 in teams if team1 != team2]


def _get_all_tournament_matches(matches: Matches) -> pd.DataFrame:
    match_pairs = matches.team_names_per_id.apply(_get_matches_one_id).explode()

    all_match_pairs = pd.DataFrame(
        match_pairs.to_list(),
        index=match_pairs.index,
        columns=["home", "away"],
    )
    return all_match_pairs.set_index(["home", "away"], append=True)


def get_double_round_robin_tournaments(matches: Matches) -> list[str]:
    """
    Returns all 'id' values corresponding to double round-robin tournaments.
    """

    all_tourney_matches = _get_all_tournament_matches(matches)

    match_count = matches.home_vs_away_count_per_id.reindex_like(all_tourney_matches)
    double_round_robins = (match_count == 1).groupby("id", observed=True).all()

    return double_round_robins[double_round_robins].index.to_list()


@dataclass
class FilterDRR:
    """
    Contains a list of 'id's of all perfect double round-robin tournaments.

    drr_ids: dict[str, list[str]]
        Maps each sport to its list of ids.
    """

    drr_ids: dict[str, list[str]]

    @classmethod
    def from_sport_to_matches(cls, sport_to_matches: dict[str, Matches]) -> FilterDRR:
        drr_ids = {
            sport: get_double_round_robin_tournaments(matches)
            for sport, matches in sport_to_matches.items()
        }

        return cls(drr_ids)

    def filter_double_round_robins(
        self,
        sport_to_df: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        Filter only double round-robin tournaments.'

        ---
        Parameters:

            sport_to_df: dict[str, pd.DataFrame]
                Maps each sport to a pd.DataFrame that should be filtered.

                "id" must be the first dataframe index level.

        ---
        Returns:
            dict[str, pd.DataFrame]
                Maps each sport to its filtered dataframe.
        """

        return {
            sport: sport_to_df[sport].loc[index]
            for sport, index in self.drr_ids.items()
        }
