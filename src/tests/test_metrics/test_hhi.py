import pandas as pd
import pytest

import turning_point.metrics.normalized_hhi as hhi


@pytest.fixture()
def values():
    return [
        pd.DataFrame({"points": [40, 30, 15, 15]}),
        pd.DataFrame({"points": [20, 20, 20, 20, 20]}),
        pd.DataFrame({"points": [35, 20, 6, 4, 3, 10, 13, 9]}),
    ]


def test_hhi(values: list[pd.DataFrame]):
    expected_values = [0.29500000000000004, 0.20000000000000004, 0.2036]

    for value, expected in zip(values, expected_values):
        results = hhi.herfindahl_hirschman_index(value)
        assert results["points"] == expected


def test_nhhi(values: list[pd.DataFrame]):
    expected_values = [
        (0.29500000000000004 - 1 / 4) / (1 - 1 / 4),
        (0.20000000000000004 - 1 / 5) / (1 - 1 / 5),
        (0.2036 - 1 / 8) / (1 - 1 / 8),
    ]

    for value, expected in zip(values, expected_values):
        results = hhi.normalized_herfindahl_hirschman_index(value)
        assert results["points"] == expected


def test_nicb(values: list[pd.DataFrame]):
    expected_values = [
        0.29500000000000004 / (1 / 4),
        0.20000000000000004 / (1 / 5),
        0.2036 / (1 / 8),
    ]

    for value, expected in zip(values, expected_values):
        results = hhi.herfindahl_index_of_competitive_balance(value)
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


def test_calculate_normalized_hhi_per_id(dataframe: pd.DataFrame):
    expected_cols = {
        "id": ["1", "2", "3"],
        "points": [
            hhi.normalized_herfindahl_hirschman_index(pd.Series([6, 0, 0])),
            hhi.normalized_herfindahl_hirschman_index(pd.Series([2, 2, 3, 1])),
            hhi.normalized_herfindahl_hirschman_index(pd.Series([0, 3])),
        ],
    }
    expected = pd.DataFrame(data=expected_cols).set_index("id")

    assert hhi._calculate_normalized_hhi_per_id(dataframe).equals(expected)


def test_calculate_hicb_per_id(dataframe: pd.DataFrame):
    expected_cols = {
        "id": ["1", "2", "3"],
        "points": [
            hhi.herfindahl_index_of_competitive_balance(pd.Series([6, 0, 0])),
            hhi.herfindahl_index_of_competitive_balance(pd.Series([2, 2, 3, 1])),
            hhi.herfindahl_index_of_competitive_balance(pd.Series([0, 3])),
        ],
    }
    expected = pd.DataFrame(data=expected_cols).set_index("id")

    assert hhi._calculate_hicb_per_id(dataframe).equals(expected)
