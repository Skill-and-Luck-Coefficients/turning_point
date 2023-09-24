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
    rankings = PointsPerMatch.from_home_away_winner(matches.home_away_winner()).rankings

    by = ["id", "points", "team"]
    sorted_rankings = rankings.sort_values(by, ascending=[True, False, True])
    return _aggregate_teams_names_per_id(sorted_rankings)


def _extract_groups(index: pd.Index, pattern: str) -> pd.Index:
    return index.str.extract(pattern, expand=False)


# TODO: Fazer mapeamento manual -> torneios podem ter times rebaixados e promovidos!
def _get_previous_ordering(
    id_yesterday_teams: pd.Series, id_to_team_names: pd.Series
) -> list[str]:
    """
    Yesterday teams: teams in last year's tourney
    Current teams: teams in this year's tourney

    Intersection: sorted by yesterday teams
        That is, last year's ranking will determine best teams
    New teams (current - yesterday): sorted alphabetically
        Put at the end:
            Assumed that: (a) teams got promoted; and (b) they are worse than the others
    """
    id = str(id_yesterday_teams.name)  # conversion just for IDE (it already is a str)
    yesterday_teams = id_yesterday_teams["yesterday teams"]
    current_teams = id_to_team_names.loc[id]

    kept_teams = [team for team in yesterday_teams if team in current_teams]
    new_teams = [team for team in sorted(current_teams) if team not in yesterday_teams]
    return kept_teams + new_teams


def from_previous_rankings(
    matches: Matches, tourney_pattern: str = ID_EXCEPT_YEAR
) -> pd.Series:
    """
    For each id, get teams sorted by its last year's ranking.

    ----
    Parameters:
        matches: Matches
            Tournament matches

        tourney_pattern: str = ID_EXCEPT_YEAR
            Regex pattern for grouping different years of the same tournament together.
    """
    current_teams = from_current_rankings(matches)

    # Maybe tournaments groupbpy should be a parameter
    same_tourney = _extract_groups(current_teams.index, tourney_pattern)
    yesterday_teams: pd.Series = current_teams.groupby(same_tourney).shift(1).dropna()

    yesterday_teams_df = yesterday_teams.to_frame("yesterday teams")
    return yesterday_teams_df.apply(
        _get_previous_ordering,
        id_to_team_names=current_teams,
        axis="columns",
    )
