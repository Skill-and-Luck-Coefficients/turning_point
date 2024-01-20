import numpy as np
import pandas as pd

import tournament_simulations.data_structures as ds
import turning_point.metrics.variances as var


def test_calculate_ranking_variances_per_id():
    test_cols = {
        "id": ["1", "1", "1", "1", "2", "2", "2", "2", "2", "2", "2", "2", "3", "3"],
        "team": ["A", "B", "A", "C", "a", "b", "c", "d", "c", "b", "c", "a", "d", "b"],
        "points": [3, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3],
        "date number": [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0],
    }
    test = pd.DataFrame(data=test_cols).set_index(["id", "date number"])

    expected_cols = {
        "id": ["1", "2", "3"],
        "points": [
            np.var([6, 0, 0], ddof=1),
            np.var([2, 2, 3, 1], ddof=1),
            np.var([0, 3], ddof=1),
        ],
    }
    expected = pd.DataFrame(data=expected_cols).set_index("id")

    assert var._calculate_ranking_variances_per_id(test).equals(expected)
