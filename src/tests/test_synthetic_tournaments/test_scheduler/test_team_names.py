import pandas as pd
import pytest

import synthetic_tournaments.scheduler.team_names as tn
from tournament_simulations.data_structures import Matches, PointsPerMatch


@pytest.fixture()
def matches():
    previous_id = "0@/sport/name/year-0"
    current_id = "0@/sport/name/year-1"

    first_portion = pd.DataFrame(
        {
            "id": [previous_id] * 2 + [current_id] * 4,
            "date number": [0, 0, 0, 1, 1, 2],
            "home": ["B", "B", "A", "A", "C", "A"],
            "away": ["A", "A", "B", "B", "B", "B"],
            "winner": ["h", "h", "a", "a", "d", "h"],
        }
    )

    previous_id = "1@/sport/name/year-0"
    current_id = "1@/sport/name/year-1"

    second_portion = pd.DataFrame(
        {
            "id": [previous_id] * 2 + [current_id] * 4,
            "date number": [0, 0, 0, 1, 0, 0],
            "home": ["a", "b", "a", "c", "b", "c"],
            "away": ["b", "a", "b", "b", "d", "d"],
            "winner": ["a", "h", "a", "a", "d", "h"],
        }
    )

    return Matches(pd.concat([first_portion, second_portion]))


@pytest.fixture
def rankings(matches: Matches):
    ppm = PointsPerMatch.from_home_away_winner(matches.home_away_winner)
    return ppm.rankings


def test_aggregate_teams_names_per_id(rankings):
    expected = pd.Series(
        index=[
            "0@/sport/name/year-0",
            "0@/sport/name/year-1",
            "1@/sport/name/year-0",
            "1@/sport/name/year-1",
        ],
        data=[
            ["A", "B"],
            ["A", "B", "C"],
            ["a", "b"],
            ["a", "b", "c", "d"],
        ],
    )
    assert tn._aggregate_teams_names_per_id(rankings).equals(expected)


def test_from_current_rankings(matches):
    expected = pd.Series(
        index=[
            "0@/sport/name/year-0",
            "0@/sport/name/year-1",
            "1@/sport/name/year-0",
            "1@/sport/name/year-1",
        ],
        data=[
            ["B", "A"],
            ["B", "A", "C"],
            ["b", "a"],
            ["b", "c", "d", "a"],
        ],
    )
    assert tn.from_current_rankings(matches).equals(expected)


def test_extract_groups(matches: Matches):
    expected = pd.Index(
        [
            "0@/sport/name/",
            "0@/sport/name/",
            "0@/sport/name/",
            "0@/sport/name/",
            "0@/sport/name/",
            "0@/sport/name/",
            "1@/sport/name/",
            "1@/sport/name/",
            "1@/sport/name/",
            "1@/sport/name/",
            "1@/sport/name/",
            "1@/sport/name/",
        ]
    )

    matches_index = matches.df.index.get_level_values("id")
    no_year = tn._extract_groups(matches_index, r"(.+?@/.+?/.+?/).+")
    assert no_year.equals(expected)


def test_get_previous_ordering():
    result = tn._get_previous_ordering(
        pd.Series(index=["yesterday teams"], data=[["C", "B", "A"]], name="name"),
        pd.Series(index=["name"], data=[["B", "C", "A"]]),
    )
    assert result == ["C", "B", "A"]

    result = tn._get_previous_ordering(
        pd.Series(index=["yesterday teams"], data=[["B", "A"]], name="name"),
        pd.Series(index=["name"], data=[["B", "C", "A"]]),
    )
    assert result == ["B", "A", "C"]


def test_from_previous_rankings(matches: Matches):
    expected = pd.Series(
        index=["0@/sport/name/year-1", "1@/sport/name/year-1"],
        data=[["B", "A", "C"], ["b", "a", "c", "d"]],
    )

    assert tn.from_previous_rankings(matches).equals(expected)
