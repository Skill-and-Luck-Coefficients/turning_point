import pandas as pd

import turning_point.variance_stats.calculate_expanding_variance_stats as cevs
from tournament_simulations.data_structures import Matches


def test_expanding_template():

    cols = {
        "id": ["1", "1", "1"],
        "date number": [0, 1, 2],
        "home": ["a", "A", "B"],
        "away": ["b", "C", "D"],
        "winner": ["h", "d", "d"],
    }

    matches = Matches(pd.DataFrame(cols))

    expected = pd.DataFrame(
        {
            "id": pd.Categorical(["1", "1", "1", "1", "1", "1"]),
            "date number": [0, 0, 1, 0, 1, 2],
            "home": pd.Categorical(["a", "a", "A", "a", "A", "B"]),
            "away": pd.Categorical(["b", "b", "C", "b", "C", "D"]),
            "winner": ["h", "h", "d", "h", "d", "d"],
            "final date": [0, 1, 1, 2, 2, 2],
        }
    ).set_index(["id", "date number"])

    assert (
        cevs._expanding_template(matches, lambda x: x.df)
        .astype({"final date": int})
        .equals(expected)
    )

    cols = {
        "id": ["1", "2", "2"],
        "date number": [0, 0, 1],
        "home": ["a", "A", "B"],
        "away": ["b", "C", "D"],
        "winner": ["h", "d", "d"],
    }

    matches = Matches(pd.DataFrame(cols))

    expected = pd.DataFrame(
        {
            "id": pd.Categorical(["1", "2", "2", "2"]),
            "date number": [0, 0, 0, 1],
            "home": pd.Categorical(["a", "A", "A", "B"]),
            "away": pd.Categorical(["b", "C", "C", "D"]),
            "winner": ["h", "d", "d", "d"],
            "final date": [0, 0, 1, 1],
        }
    ).set_index(["id", "date number"])

    assert (
        cevs._expanding_template(matches, lambda x: x.df)
        .astype({"final date": int})
        .equals(expected)
    )
