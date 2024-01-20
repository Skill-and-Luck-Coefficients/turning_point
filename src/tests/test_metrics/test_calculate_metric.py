import numpy as np
import pandas as pd

import tournament_simulations.data_structures as ds
import turning_point.metrics.calculate_metric as cmet
from turning_point.metrics.variances import _calculate_ranking_variances_per_id as f


def test_get_kwargs_from_points_per_match_id_to_probabilities():
    test_cols = {
        "id": ["1", "1", "1", "1", "2", "2", "2", "2"],
        "team": ["A", "B", "A", "C", "a", "b", "c", "d"],
        "points": [3, 0, 3, 0, 1, 1, 1, 1],
        "date number": [0, 0, 1, 1, 0, 0, 0, 0],
    }
    test = ds.PointsPerMatch(
        pd.DataFrame(data=test_cols).set_index(["id", "date number"])
    )

    expected_real_cols = {
        "id": pd.Categorical(["1", "2"]),
        "real": [np.var([6, 0, 0], ddof=1), np.var([1, 1, 1, 1], ddof=1)],
    }
    expected_real = pd.DataFrame(expected_real_cols).set_index("id")

    expected_simul_cols = {
        "id": pd.Categorical(["1", "2"]),
        "s0": [np.var([0, 3, 3], ddof=1), np.var([3, 0, 3, 0], ddof=1)],
        "s1": [np.var([0, 3, 3], ddof=1), np.var([3, 0, 3, 0], ddof=1)],
        "s2": [np.var([0, 3, 3], ddof=1), np.var([3, 0, 3, 0], ddof=1)],
        "s3": [np.var([0, 3, 3], ddof=1), np.var([3, 0, 3, 0], ddof=1)],
        "s4": [np.var([0, 3, 3], ddof=1), np.var([3, 0, 3, 0], ddof=1)],
        "s5": [np.var([0, 3, 3], ddof=1), np.var([3, 0, 3, 0], ddof=1)],
    }
    expected_simul = pd.DataFrame(data=expected_simul_cols).set_index("id")
    id_to_prob = pd.Series(
        index=["1", "2"],
        data=[
            {(3, 0): 0, (1, 1): 0, (0, 3): 1},
            {(3, 0): 1, (1, 1): 0, (0, 3): 0},
        ],
    )

    kwargs_params = cmet.get_kwargs_from_points_per_match(test, f, (2, 3), id_to_prob)
    assert kwargs_params["simulated"].equals(expected_simul)
    assert kwargs_params["real"].equals(expected_real)

    kwargs_params = cmet.get_kwargs_from_points_per_match(test, f, (3, 2), id_to_prob)
    assert kwargs_params["simulated"].equals(expected_simul)
    assert kwargs_params["real"].equals(expected_real)


def test_get_kwargs_from_points_per_match_id_to_probabilities_new_points():
    test_cols = {
        "id": ["1", "1", "1", "1", "2", "2", "2", "2"],
        "team": ["A", "B", "A", "C", "a", "b", "c", "d"],
        "points": [2, 1, 2, 1, 1, 2, 1, 2],
        "date number": [0, 0, 1, 1, 0, 0, 0, 0],
    }
    test = ds.PointsPerMatch(
        pd.DataFrame(data=test_cols).set_index(["id", "date number"])
    )

    expected_real_cols = {
        "id": pd.Categorical(["1", "2"]),
        "real": [np.var([4, 1, 1], ddof=1), np.var([1, 2, 1, 2], ddof=1)],
    }
    expected_real = pd.DataFrame(expected_real_cols).set_index("id")

    expected_simul_cols = {
        "id": pd.Categorical(["1", "2"]),
        "s0": [np.var([2, 2, 2], ddof=1), np.var([0, 0, 0, 0], ddof=1)],
        "s1": [np.var([2, 2, 2], ddof=1), np.var([0, 0, 0, 0], ddof=1)],
        "s2": [np.var([2, 2, 2], ddof=1), np.var([0, 0, 0, 0], ddof=1)],
        "s3": [np.var([2, 2, 2], ddof=1), np.var([0, 0, 0, 0], ddof=1)],
        "s4": [np.var([2, 2, 2], ddof=1), np.var([0, 0, 0, 0], ddof=1)],
        "s5": [np.var([2, 2, 2], ddof=1), np.var([0, 0, 0, 0], ddof=1)],
    }
    expected_simul = pd.DataFrame(data=expected_simul_cols).set_index("id")
    id_to_prob = pd.Series(
        index=["1", "2"],
        data=[
            {(2, 1): 0, (0, 0): 0, (1, 2): 1},
            {(2, 1): 0, (0, 0): 1, (1, 2): 0},
        ],
    )

    kwargs_params = cmet.get_kwargs_from_points_per_match(test, f, (2, 3), id_to_prob)
    assert kwargs_params["simulated"].equals(expected_simul)
    assert kwargs_params["real"].equals(expected_real)

    kwargs_params = cmet.get_kwargs_from_points_per_match(test, f, (3, 2), id_to_prob)
    assert kwargs_params["simulated"].equals(expected_simul)
    assert kwargs_params["real"].equals(expected_real)


def test_get_kwargs_from_points_per_match_first():
    test_cols = {
        "id": ["1", "1", "1", "1", "2", "2", "2", "2"],
        "team": ["A", "B", "A", "C", "a", "b", "c", "d"],
        "points": [3, 0, 3, 0, 1, 1, 1, 1],
        "date number": [0, 0, 1, 1, 0, 0, 0, 0],
    }
    test = ds.PointsPerMatch(
        pd.DataFrame(data=test_cols).set_index(["id", "date number"])
    )

    expected_real_cols = {
        "id": pd.Categorical(["1", "2"]),
        "real": [np.var([6, 0, 0], ddof=1), np.var([1, 1, 1, 1], ddof=1)],
    }
    expected_real = pd.DataFrame(expected_real_cols).set_index("id")

    expected_simul_cols = {
        "id": pd.Categorical(["1", "2"]),
        "s0": [np.var([6, 0, 0], ddof=1), np.var([1, 1, 1, 1], ddof=1)],
        "s1": [np.var([6, 0, 0], ddof=1), np.var([1, 1, 1, 1], ddof=1)],
        "s2": [np.var([6, 0, 0], ddof=1), np.var([1, 1, 1, 1], ddof=1)],
        "s3": [np.var([6, 0, 0], ddof=1), np.var([1, 1, 1, 1], ddof=1)],
        "s4": [np.var([6, 0, 0], ddof=1), np.var([1, 1, 1, 1], ddof=1)],
        "s5": [np.var([6, 0, 0], ddof=1), np.var([1, 1, 1, 1], ddof=1)],
    }
    expected_simul = pd.DataFrame(data=expected_simul_cols).set_index("id")

    kwargs_params = cmet.get_kwargs_from_points_per_match(test, f, (2, 3))
    assert kwargs_params["simulated"].equals(expected_simul)
    assert kwargs_params["real"].equals(expected_real)

    kwargs_params = cmet.get_kwargs_from_points_per_match(test, f, (3, 2))
    assert kwargs_params["simulated"].equals(expected_simul)
    assert kwargs_params["real"].equals(expected_real)


def test_get_kwargs_from_points_per_match_second():
    test_cols = {
        "id": ["1", "1", "2", "2", "3", "3", "4", "4"],
        "team": ["A", "B", "A", "C", "a", "b", "c", "d"],
        "points": [3, 0, 0, 3, 1, 1, 1, 1],
        "date number": [0, 0, 0, 0, 0, 0, 0, 0],
    }
    test = ds.PointsPerMatch(
        pd.DataFrame(data=test_cols).set_index(["id", "date number"])
    )

    expected_real_cols = {
        "id": pd.Categorical(["1", "2", "3", "4"]),
        "real": [
            np.var([3, 0], ddof=1),
            np.var([0, 3], ddof=1),
            np.var([1, 1], ddof=1),
            np.var([1, 1], ddof=1),
        ],
    }
    expected_real = pd.DataFrame(data=expected_real_cols).set_index("id")

    expected_simul_cols = {
        "id": pd.Categorical(["1", "2", "3", "4"]),
        "s0": [
            np.var([3, 0], ddof=1),
            np.var([0, 3], ddof=1),
            np.var([1, 1], ddof=1),
            np.var([1, 1], ddof=1),
        ],
        "s1": [
            np.var([3, 0], ddof=1),
            np.var([0, 3], ddof=1),
            np.var([1, 1], ddof=1),
            np.var([1, 1], ddof=1),
        ],
        "s2": [
            np.var([3, 0], ddof=1),
            np.var([0, 3], ddof=1),
            np.var([1, 1], ddof=1),
            np.var([1, 1], ddof=1),
        ],
        "s3": [
            np.var([3, 0], ddof=1),
            np.var([0, 3], ddof=1),
            np.var([1, 1], ddof=1),
            np.var([1, 1], ddof=1),
        ],
    }
    expected_simul = pd.DataFrame(data=expected_simul_cols).set_index("id")

    kwargs_params = cmet.get_kwargs_from_points_per_match(test, f, (4, 1))
    assert kwargs_params["simulated"].equals(expected_simul)
    assert kwargs_params["real"].equals(expected_real)

    kwargs_params = cmet.get_kwargs_from_points_per_match(test, f, (1, 4))
    assert kwargs_params["simulated"].equals(expected_simul)
    assert kwargs_params["real"].equals(expected_real)
