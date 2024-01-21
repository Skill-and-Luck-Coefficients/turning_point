import pandas as pd
import pytest

import turning_point.metrics.gini_index as gini


@pytest.fixture()
def values():
    return [
        pd.DataFrame({"points": [1, 1, 1]}),
        pd.DataFrame({"points": [2e4, 3e4, 4e4, 5e4, 6e4]}),
        pd.DataFrame({"points": [9e3, 4e4, 48e3, 48e3, 55e3]}),
        pd.DataFrame({"points": [2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 12e4, 15e4]}),
        pd.DataFrame(
            {"points": [5e4, 5e4, 9e4, 9e4, 13e4, 13e4, 17e4, 17e4, 27e4, 27e4]}
        ),
    ]


def test_gini(values: list[pd.DataFrame]):
    series = pd.Series([2e4, 3e4, 4e4, 5e4, 6e4])
    assert gini.gini(series) == 0.2

    df = pd.DataFrame({"points": [2e4, 3e4, 4e4, 5e4, 6e4], "ones": [1] * 5})
    result = gini.gini(df)
    assert result["points"] == 0.2
    assert result["ones"] == 0

    expected_values = [0, 0.2, 0.2, 0.3028169014084507, 0.29295774647887324]

    for value, expected in zip(values, expected_values):
        results = gini.gini(value)
        assert results["points"] == expected


@pytest.fixture()
def dataframe():
    df_cols = {
        "id": ["1", "1", "1", "1", "2", "2", "2", "2", "2", "2", "2", "2", "3", "3"],
        "team": ["A", "B", "A", "C", "a", "b", "c", "d", "c", "b", "c", "a", "d", "b"],
        "points": [3, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3],
        "date number": [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0],
    }
    return pd.DataFrame(data=df_cols).set_index(["id", "date number"])


def test_calculate_gini_index_per_id(dataframe: pd.DataFrame):
    expected_cols = {
        "id": ["1", "2", "3"],
        "points": [
            gini.gini(pd.Series([6, 0, 0])),
            gini.gini(pd.Series([2, 2, 3, 1])),
            gini.gini(pd.Series([0, 3])),
        ],
    }
    expected = pd.DataFrame(data=expected_cols).set_index("id")

    assert gini._calculate_gini_index_per_id(dataframe).equals(expected)
