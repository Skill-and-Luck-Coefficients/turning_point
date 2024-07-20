import numpy as np
import pandas as pd

from synthetic_tournaments.bradley_terry import create_schedule as cs


def test_replace_team_name_with_strength():
    matches = pd.DataFrame(
        {
            "home": [0, 1, 0, 2],
            "away": [2, 0, 1, 0],
        }
    )
    strengths = [0.5, 0.25, 1]

    expected = {
        "home": pd.Series([0.5, 0.25, 0.5, 1]),
        "away": pd.Series([1, 0.5, 0.25, 0.5]),
    }

    result = cs._replace_team_name_with_strength(matches, strengths)
    assert result["home"].equals(expected["home"])
    assert result["away"].equals(expected["away"])


def test_get_bradley_terry_winner():
    skill_per_match = {
        "home": pd.Series([0.5, 0.25, 0.5, 1]),
        "away": pd.Series([1, 0.5, 0.25, 0.5]),
    }
    uniform_values_list = [
        np.array([0, 0, 0, 0]),
        np.array([1, 1, 1, 1]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.5, 0.3, 0.7, 0.5]),
    ]

    expected_list = [
        pd.Series(["h", "h", "h", "h"]),
        pd.Series(["a", "a", "a", "a"]),
        pd.Series(["a", "a", "h", "h"]),
        pd.Series(["a", "h", "a", "h"]),
    ]

    for values, expected in zip(uniform_values_list, expected_list):
        result = cs._get_bradley_terry_winner(skill_per_match, values)
        assert result.equals(expected)


def test_simulate_bt_tourney_no_randomness():
    random_fn = lambda size: np.array([1] * size)
    rand_first = None

    expected = pd.DataFrame(
        {
            "id": pd.Categorical(["tourney"] * 6),
            "date number": [0, 1, 2, 3, 4, 5],
            "home": pd.Categorical([0, 2, 1, 1, 0, 2]),
            "away": pd.Categorical([1, 0, 2, 0, 2, 1]),
            "winner": ["a"] * 6,
        }
    ).set_index(["id", "date number"])

    args = ([1, 0.5, 0.25], "tourney", 1)
    result = cs._simulate_bt_tourney_no_randomness(*args, random_fn, rand_first)
    assert result.df.equals(expected)

    expected["winner"] = "h"
    random_fn = lambda size: np.array([0] * size)

    result = cs._simulate_bt_tourney_no_randomness(*args, random_fn, rand_first)
    assert result.df.equals(expected)
