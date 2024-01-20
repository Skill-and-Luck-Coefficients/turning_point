import numpy as np
import pandas as pd

import turning_point.normal_coefficient.calculate_turning_point as ctp
from turning_point.metric_stats import ExpandingMetricStats


def test_find_turning_point_one_id():
    iter = []
    assert np.isnan(ctp._find_turning_point_one_id(iter))

    iter = [True, True]
    assert ctp._find_turning_point_one_id(iter) == 0

    iter = (True, False, False)
    assert np.isinf(ctp._find_turning_point_one_id(iter))

    iter = [True, False, True, True]
    assert ctp._find_turning_point_one_id(iter) == 2

    iter = [True, False, False, True, True]
    assert ctp._find_turning_point_one_id(iter) == 3

    iter = [True, False, False, True, False, False, True]
    assert ctp._find_turning_point_one_id(iter) == 6


def test_find_turning_point_percent_one_id():
    iter = []
    assert np.isnan(ctp._find_turning_point_percent_one_id(iter))

    iter = [True, True]
    assert ctp._find_turning_point_percent_one_id(iter) == 1 / 2

    iter = (True, False, False)
    assert np.isinf(ctp._find_turning_point_percent_one_id(iter))

    iter = [True, False, True, True]
    assert ctp._find_turning_point_percent_one_id(iter) == 3 / 4

    iter = [True, False, False, True, True]
    assert ctp._find_turning_point_percent_one_id(iter) == 4 / 5

    iter = [True, False, False, True, False, False, True]
    assert ctp._find_turning_point_percent_one_id(iter) == 1


def test_get_kwargs_from_expanding_variances_stats():
    var_quantile = ExpandingMetricStats(
        pd.DataFrame(
            {
                "id": pd.Categorical(["1", "1", "2", "2", "2"]),
                "final date": [0, 1, 0, 1, 2],
                "real": [1, 3, 2, 2, 5],
                "mean": [0, 1, 2, 3, 4],
                "quantile": [0, 2, 1, 3, 4],
            }
        ).set_index(["id", "final date"])
    )

    expected_cols = {
        "id": pd.Categorical(["1", "2"]),
        "turning point": [0, 2],
        "%turning point": [0.5, 1],
    }
    expected = pd.DataFrame(expected_cols).set_index("id")

    turning_point = ctp.get_kwargs_from_expanding_variances_stats(var_quantile)["df"]
    assert turning_point.equals(expected)
