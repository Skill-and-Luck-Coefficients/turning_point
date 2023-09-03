import pandas as pd
import pytest

import synthetic_tournaments.scheduler.num_schedules as ns
from tournament_simulations.data_structures import Matches


@pytest.fixture()
def first_matches():
    previous_id = "0@/sport/name/year-0"
    current_id = "0@/sport/name/year-1"

    return Matches(
        pd.DataFrame(
            {
                "id": [previous_id] * 2 + [current_id] * 4,
                "date number": [0, 0, 0, 1, 1, 2],
                "home": ["B", "B", "A", "A", "C", "A"],
                "away": ["A", "A", "B", "B", "B", "B"],
                "winner": ["h", "h", "a", "a", "d", "h"],
            }
        )
    )


@pytest.fixture()
def second_matches():
    previous_id = "1@/sport/name/year-0"
    current_id = "1@/sport/name/year-1"

    return Matches(
        pd.DataFrame(
            {
                "id": [previous_id] * 2 + [current_id] * 4,
                "date number": [0, 0, 0, 1, 0, 0],
                "home": ["a", "b", "a", "c", "b", "c"],
                "away": ["b", "a", "b", "b", "d", "d"],
                "winner": ["a", "h", "a", "a", "d", "h"],
            }
        )
    )


def test_from_matches_first(first_matches):
    previous_id = "0@/sport/name/year-0"
    current_id = "0@/sport/name/year-1"

    expected = pd.Series(index=[previous_id, current_id], data=[2, 3])
    assert ns.from_matches(first_matches).equals(expected)


def test_from_matches_second(second_matches):
    previous_id = "1@/sport/name/year-0"
    current_id = "1@/sport/name/year-1"

    expected = pd.Series(index=[previous_id, current_id], data=[1, 1])
    assert ns.from_matches(second_matches).equals(expected)
