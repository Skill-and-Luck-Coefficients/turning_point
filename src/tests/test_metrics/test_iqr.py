import pandas as pd
import pytest

import turning_point.metrics.interquartile_range as iqr


@pytest.fixture()
def values():
    return [
        pd.DataFrame(
            {"points": [7, 7, 31, 31, 47, 75, 87, 115, 116, 119, 119, 155, 177]}
        ),
        pd.DataFrame(
            {
                "points": [
                    11,
                    13,
                    16,
                    19,
                    20,
                    21,
                    23,
                    25,
                    26,
                    29,
                    33,
                    34,
                    36,
                    38,
                    39,
                    46,
                    52,
                    55,
                    58,
                ]
            }
        ),
        pd.DataFrame({"points": [45, 47, 52, 52, 53, 55, 56, 58, 62, 80]}),
    ]


def test_gini(values: list[pd.DataFrame]):
    expected_values = [88, 18, 5.5]

    for value, expected in zip(values, expected_values):
        results = iqr.interquartile_range(value)
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
            iqr.interquartile_range(pd.Series([6, 0, 0])),
            iqr.interquartile_range(pd.Series([2, 2, 3, 1])),
            iqr.interquartile_range(pd.Series([0, 3])),
        ],
    }
    expected = pd.DataFrame(data=expected_cols).set_index("id")

    assert iqr._calculate_interquartile_range_per_id(dataframe).equals(expected)
