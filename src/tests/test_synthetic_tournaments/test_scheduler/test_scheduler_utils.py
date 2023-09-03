import pandas as pd

import synthetic_tournaments.scheduler.utils as utils


def test_agg_tuple_per_id():
    first_series = pd.Series(index=["0", "1"], data=[["a", "b"], ["C", "D"]])
    second_series = pd.Series(index=["0", "1"], data=[1, 2])
    to_agg = [first_series, second_series]

    expected = pd.Series(index=["0", "1"], data=[(["a", "b"], 1), (["C", "D"], 2)])
    assert utils.agg_tuple_per_id(to_agg).equals(expected)

    third_series = pd.Series(index=["0"], data=[0])
    to_agg = [first_series, second_series, third_series]

    expected = pd.Series(index=["0"], data=[(["a", "b"], 1, 0)])
    assert utils.agg_tuple_per_id(to_agg).equals(expected)
