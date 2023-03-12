import numpy as np
import pandas as pd
import pytest

import turning_point.match_coefficient.create_match_turning_point as cmtp
from tournament_simulations.data_structures import Matches
from turning_point.normal_coefficient import TurningPoint


@pytest.fixture
def matches():

    df = pd.DataFrame(
        {
            "id": ["2", "2", "2", "1", "1", "1"],
            "date number": [0, 0, 1, 0, 1, 1],
            "home": ["a", "b", "a", "A", "A", "C"],
            "away": ["b", "c", "c", "B", "C", "B"],
            "winner": ["a", "a", "h", "d", "d", "h"],
        }
    )
    return Matches(df)


@pytest.fixture
def turning_point():

    df = pd.DataFrame({"id": ["2", "1", "3"], "turning point": [1, 2, np.nan]})
    return TurningPoint(df.set_index("id"))


@pytest.fixture
def turning_point_col():

    return [("1", 2), ("2", 1), ("3", np.nan)]


def test_count_number_and_percentage_of_matches_in_interval(
    matches: Matches,
    turning_point_col: list[tuple[str, float]],
):

    nan = cmtp._count_number_and_percentage_of_matches_in_interval(
        turning_point_col[2], matches
    )
    assert np.isnan(nan[0])
    assert np.isnan(nan[1])

    expected = (2, 2 / 3)

    assert (
        cmtp._count_number_and_percentage_of_matches_in_interval(
            turning_point_col[1], matches
        )
        == expected
    )

    expected = (3, 3 / 3)

    assert (
        cmtp._count_number_and_percentage_of_matches_in_interval(
            turning_point_col[0], matches
        )
        == expected
    )


def test_get_kwargs_from_matches_turning_point(
    matches: Matches, turning_point: TurningPoint
):

    expected = (
        pd.DataFrame(
            {
                "id": pd.Categorical(["2", "1", "3"]),
                "match turning point": [2, 3, np.nan],
                "%match turning point": [2 / 3, 1, np.nan],
            }
        )
        .set_index("id")
        .sort_index()
    )

    match_tp = cmtp.get_kwargs_from_matches_turning_point(matches, turning_point)
    assert match_tp["df"].equals(expected)
