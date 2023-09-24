import numpy as np
import pandas as pd

import tournament_simulations.data_structures as ds
import turning_point.variances.calculate_variances as var


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
        "real var": [np.var([6, 0, 0], ddof=1), np.var([1, 1, 1, 1], ddof=1)],
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

    kwargs_params = var.get_kwargs_from_points_per_match(test, (2, 3), id_to_prob)
    assert kwargs_params["simulated"].equals(expected_simul)
    assert kwargs_params["real"].equals(expected_real)

    kwargs_params = var.get_kwargs_from_points_per_match(test, (3, 2), id_to_prob)
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
        "real var": [np.var([4, 1, 1], ddof=1), np.var([1, 2, 1, 2], ddof=1)],
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

    kwargs_params = var.get_kwargs_from_points_per_match(test, (2, 3), id_to_prob)
    assert kwargs_params["simulated"].equals(expected_simul)
    assert kwargs_params["real"].equals(expected_real)

    kwargs_params = var.get_kwargs_from_points_per_match(test, (3, 2), id_to_prob)
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
        "real var": [np.var([6, 0, 0], ddof=1), np.var([1, 1, 1, 1], ddof=1)],
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

    kwargs_params = var.get_kwargs_from_points_per_match(test, (2, 3))
    assert kwargs_params["simulated"].equals(expected_simul)
    assert kwargs_params["real"].equals(expected_real)

    kwargs_params = var.get_kwargs_from_points_per_match(test, (3, 2))
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
        "real var": [
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

    kwargs_params = var.get_kwargs_from_points_per_match(test, (4, 1))
    assert kwargs_params["simulated"].equals(expected_simul)
    assert kwargs_params["real"].equals(expected_real)

    kwargs_params = var.get_kwargs_from_points_per_match(test, (1, 4))
    assert kwargs_params["simulated"].equals(expected_simul)
    assert kwargs_params["real"].equals(expected_real)
