import numpy as np
import pandas as pd
import pytest

import turning_point.variance_stats.get_matches_in_window as gmiw
from tournament_simulations.data_structures import Matches


@pytest.fixture
def matches_fixture():

    cols = {
        "id": ["1", "1", "1", "1", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2"],
        "date number": [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        "home": ["0", "0", "1", "1", "0", "0", "0", "0", "1", "1", "2", "2", "3", "3"],
        "away": ["0", "0", "1", "1", "0", "0", "0", "0", "1", "1", "2", "2", "3", "3"],
    }
    return Matches(pd.DataFrame(data=cols).set_index(["id", "date number"]))


def test_get_mask_inside_interval(matches_fixture: Matches):

    assert np.all(
        gmiw._get_mask_inside_interval(matches_fixture, 0, 3) == np.array([True] * 14)
    )

    expected = [True] * 2 + [False] * 2 + [True] * 4 + [False] * 6
    assert np.all(
        gmiw._get_mask_inside_interval(matches_fixture, 0, 0) == np.array(expected)
    )

    expected = [True] * 12 + [False] * 2
    assert np.all(
        gmiw._get_mask_inside_interval(matches_fixture, 0, 2) == np.array(expected)
    )

    expected = [False] * 2 + [True] * 2 + [False] * 4 + [True] * 4 + [False] * 2
    assert np.all(
        gmiw._get_mask_inside_interval(matches_fixture, 1, 2) == np.array(expected)
    )

    expected = [False] * 10 + [True] * 4
    assert np.all(
        gmiw._get_mask_inside_interval(matches_fixture, 2, 5) == np.array(expected)
    )


def test_get_mask_tournaments_not_already_finished(matches_fixture: Matches):

    assert np.all(
        gmiw._get_mask_tournaments_not_already_finished(matches_fixture, 0)
        == np.array([True] * 14)
    )

    assert np.all(
        gmiw._get_mask_tournaments_not_already_finished(matches_fixture, 1)
        == np.array([True] * 14)
    )

    assert np.all(
        gmiw._get_mask_tournaments_not_already_finished(matches_fixture, 2)
        == np.array([False] * 4 + [True] * 10)
    )

    assert np.all(
        gmiw._get_mask_tournaments_not_already_finished(matches_fixture, 3)
        == np.array([False] * 4 + [True] * 10)
    )

    assert np.all(
        gmiw._get_mask_tournaments_not_already_finished(matches_fixture, 4)
        == np.array([False] * 14)
    )


def test_select_matches_inside_window(matches_fixture: Matches):

    expected_cols = {
        "id": pd.Categorical(["1", "1", "2", "2", "2", "2"]),
        "date number": [0, 0, 0, 0, 0, 0],
        "home": pd.Categorical(
            ["0", "0", "0", "0", "0", "0"], categories=["0", "1", "2", "3"]
        ),
        "away": pd.Categorical(
            ["0", "0", "0", "0", "0", "0"], categories=["0", "1", "2", "3"]
        ),
    }
    expected = pd.DataFrame(data=expected_cols).set_index(["id", "date number"])
    assert gmiw.select_matches_inside_window(matches_fixture, 0, 0).df.equals(expected)

    expected_cols = {
        "id": pd.Categorical(["1", "1", "1", "1", "2", "2", "2", "2", "2", "2"]),
        "date number": [0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        "home": pd.Categorical(
            ["0", "0", "1", "1", "0", "0", "0", "0", "1", "1"],
            categories=["0", "1", "2", "3"],
        ),
        "away": pd.Categorical(
            ["0", "0", "1", "1", "0", "0", "0", "0", "1", "1"],
            categories=["0", "1", "2", "3"],
        ),
    }
    expected = pd.DataFrame(data=expected_cols).set_index(["id", "date number"])
    assert gmiw.select_matches_inside_window(matches_fixture, 0, 1).df.equals(expected)

    expected_cols = {
        "id": pd.Categorical(["1", "1", "2", "2"]),
        "date number": [1, 1, 1, 1],
        "home": pd.Categorical(["1", "1", "1", "1"], categories=["0", "1", "2", "3"]),
        "away": pd.Categorical(["1", "1", "1", "1"], categories=["0", "1", "2", "3"]),
    }
    expected = pd.DataFrame(data=expected_cols).set_index(["id", "date number"])
    assert gmiw.select_matches_inside_window(matches_fixture, 1, 1).df.equals(expected)

    expected_cols = {
        "id": pd.Categorical(
            ["2", "2", "2", "2", "2", "2", "2", "2"], categories=["1", "2"]
        ),
        "date number": [0, 0, 0, 0, 1, 1, 2, 2],
        "home": pd.Categorical(
            ["0", "0", "0", "0", "1", "1", "2", "2"], categories=["0", "1", "2", "3"]
        ),
        "away": pd.Categorical(
            ["0", "0", "0", "0", "1", "1", "2", "2"], categories=["0", "1", "2", "3"]
        ),
    }
    expected = pd.DataFrame(data=expected_cols).set_index(["id", "date number"])
    assert gmiw.select_matches_inside_window(matches_fixture, 0, 2).df.equals(expected)
