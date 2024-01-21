import pandas as pd
import pytest

import turning_point.metrics.top_concentration_ratio as cr


@pytest.fixture()
def values():
    return [
        pd.DataFrame({"points": [1, 2, 3, 4]}),
        pd.DataFrame({"points": [1, 2, 3, 4, 5, 6]}),
        pd.DataFrame({"points": [1, 2, 3, 4, 5, 6, 7, 8]}),
    ]


def test_top_x_percent_concentration_ratio(values: list[pd.DataFrame]):
    expected_values = [0.4, 0.2857142857142857, 0.41666666666666663]

    for value, expected in zip(values, expected_values):
        results = cr.top_x_percent_concentration_ratio(value)
        assert results["points"] == expected

    expected_values = [0, 0.2857142857142857, 0.2222222222222222]

    for value, expected in zip(values, expected_values):
        results = cr.top_x_percent_concentration_ratio(value, x=0.2)
        assert results["points"] == expected


def test_normalized_top_x_percent_concentration_ratio(values: list[pd.DataFrame]):
    expected_values = [
        0.4 / (1 / 4),
        0.2857142857142857 / (1 / 6),
        0.41666666666666663 / (2 / 8),
    ]

    for value, expected in zip(values, expected_values):
        results = cr.normalized_top_x_percent_concentration_ratio(value)
        assert results["points"] == expected

    expected_values = [0, 0.2857142857142857 / (1 / 6), 0.2222222222222222 / (1 / 8)]

    for value, expected in zip(values, expected_values):
        results = cr.normalized_top_x_percent_concentration_ratio(value, x=0.2)
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


def test_calculate_normalized_top_x_cr_per_id(dataframe: pd.DataFrame):
    expected_cols = {
        "id": ["1", "2", "3"],
        "points": [
            cr.normalized_top_x_percent_concentration_ratio(pd.Series([6, 0, 0])),
            cr.normalized_top_x_percent_concentration_ratio(pd.Series([2, 2, 3, 1])),
            cr.normalized_top_x_percent_concentration_ratio(pd.Series([0, 3])),
        ],
    }
    expected = pd.DataFrame(data=expected_cols).set_index("id")

    assert cr._calculate_normalized_top_x_cr_per_id(dataframe).equals(expected)
