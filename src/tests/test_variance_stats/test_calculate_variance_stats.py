import numpy as np
import pandas as pd

import turning_point.variance_stats.calculate_variance_stats as vst
from tournament_simulations.data_structures import Matches
from turning_point.variances.variances import Variances


def nfp(*args):  # nfp -> ninety fiver percentile
    return np.percentile(args, 95)


def mean(*args):
    return np.mean(args)


def test_calculate_quantile__one():
    test_cols = {
        "id": pd.Categorical(["1", "1", "2", "2", "3"]),
        "s1": [1, 1, 2, 2, 2],
        "s2": [0, 1, 0, 1, 2],
        "s3": [1.2, 0.25, 0.1, 0.65, 0.98],
        "s4": [0, 0.4, 0.2, 0.4, 0.8],
    }
    test = pd.DataFrame(test_cols).set_index("id")

    expected_cols = {
        "id": ["1", "1", "2", "2", "3"],
        "mean": [
            mean(1, 0, 1.2, 0),
            mean(1, 1, 0.25, 0.4),
            mean(2, 0, 0.1, 0.2),
            mean(2, 1, 0.65, 0.4),
            mean(2, 2, 0.98, 0.8),
        ],
        "0.950-quantile": [
            nfp(1, 0, 1.2, 0),
            nfp(1, 1, 0.25, 0.4),
            nfp(2, 0, 0.1, 0.2),
            nfp(2, 1, 0.65, 0.4),
            nfp(2, 2, 0.98, 0.8),
        ],
    }

    expected = pd.DataFrame(expected_cols).astype({"id": "category"}).set_index("id")

    assert vst._calculate_mean_quantile(test, 0.95).equals(expected)


def test_calculate_quantile__two():
    test_cols = {
        "id": pd.Categorical(["1", "1", "2", "2", "2"]),
        "final date": [0, 1, 0, 1, 2],
        "s1": [1, 1, 2, 2, 2],
        "s2": [0, 1, 0, 1, 2],
        "s3": [1.2, 0.25, 0.1, 0.65, 0.98],
    }
    test = pd.DataFrame(test_cols).set_index(["id", "final date"])

    expected_cols = {
        "id": ["1", "1", "2", "2", "2"],
        "final date": [0, 1, 0, 1, 2],
        "mean": [
            mean(1, 0, 1.2),
            mean(1, 1, 0.25),
            mean(2, 0, 0.1),
            mean(2, 1, 0.65),
            mean(2, 2, 0.98),
        ],
        "0.950-quantile": [
            nfp(1, 0, 1.2),
            nfp(1, 1, 0.25),
            nfp(2, 0, 0.1),
            nfp(2, 1, 0.65),
            nfp(2, 2, 0.98),
        ],
    }

    expected = (
        pd.DataFrame(expected_cols)
        .astype({"id": "category"})
        .set_index(["id", "final date"])
    )

    assert vst._calculate_mean_quantile(test, 0.95).equals(expected)


def test_get_kwargs_from_variances():
    test_real_cols = {
        "id": pd.Categorical(["1", "2", "3", "4", "5"]),
        "real var": [0.45, 5.83, 4.29, 5.36, 9.37],
    }

    test_simul_cols = {
        "id": pd.Categorical(["1", "2", "3", "4", "5"]),
        "s1": [1, 1, 2, 2, 2],
        "s2": [0, 1, 0, 1, 2],
        "s3": [1.2, 0.25, 0.1, 0.65, 0.98],
        "s4": [0, 0.4, 0.2, 0.4, 0.8],
    }
    test = Variances(
        pd.DataFrame(test_real_cols).set_index("id"),
        pd.DataFrame(test_simul_cols).set_index("id"),
    )

    expected_cols = {
        "id": pd.Categorical(["1", "2", "3", "4", "5"]),
        "real var": [0.45, 5.83, 4.29, 5.36, 9.37],
        "mean": [
            mean(1, 0, 1.2, 0),
            mean(1, 1, 0.25, 0.4),
            mean(2, 0, 0.1, 0.2),
            mean(2, 1, 0.65, 0.4),
            mean(2, 2, 0.98, 0.8),
        ],
        "0.950-quantile": [
            nfp(1, 0, 1.2, 0),
            nfp(1, 1, 0.25, 0.4),
            nfp(2, 0, 0.1, 0.2),
            nfp(2, 1, 0.65, 0.4),
            nfp(2, 2, 0.98, 0.8),
        ],
    }

    expected = pd.DataFrame(expected_cols).set_index("id")

    assert vst.get_kwargs_from_variances(test, 0.95)["df"].equals(expected)


def test_get_kwargs_from_matches():
    test_cols = {
        "id": pd.Categorical(["1", "1", "2", "2", "2"]),
        "date number": [0, 0, 0, 0, 0],
        "home": pd.Categorical(["A", "A", "a", "b", "a"]),
        "away": pd.Categorical(["B", "B", "b", "c", "d"]),
        "winner": ["a", "a", "d", "d", "d"],
    }
    test = Matches(pd.DataFrame(test_cols).set_index(["id", "date number"]))

    first_id_vars = [
        np.var([0, 6], ddof=1),
        np.var([0, 6], ddof=1),
        np.var([0, 6], ddof=1),
        np.var([0, 6], ddof=1),
        np.var([0, 6], ddof=1),
        np.var([0, 6], ddof=1),
    ]

    second_id_vars = [
        np.var([2, 2, 1, 1], ddof=1),
        np.var([2, 2, 1, 1], ddof=1),
        np.var([2, 2, 1, 1], ddof=1),
        np.var([2, 2, 1, 1], ddof=1),
        np.var([2, 2, 1, 1], ddof=1),
        np.var([2, 2, 1, 1], ddof=1),
    ]

    expected = pd.DataFrame(
        {
            "id": pd.Categorical(["1", "2"]),
            "real var": [np.var([0, 6], ddof=1), np.var([2, 2, 1, 1], ddof=1)],
            "mean": [mean(*first_id_vars), mean(*second_id_vars)],
            "0.950-quantile": [nfp(*first_id_vars), nfp(*second_id_vars)],
        }
    ).set_index("id")

    result = vst.get_kwargs_from_matches(
        test,
        num_iteration_simulation=(2, 3),
        winner_type="winner",
        winner_to_points={"h": (3, 0), "d": (1, 1), "a": (0, 3)},
    )
    assert result["df"].equals(expected)
    result = vst.get_kwargs_from_matches(
        test,
        num_iteration_simulation=(3, 2),
        winner_type="winner",
        winner_to_points={"h": (3, 0), "d": (1, 1), "a": (0, 3)},
    )
    assert result["df"].equals(expected)


def test_get_kwargs_from_matches_bigger_in_to_probabilities():
    test_cols = {
        "id": pd.Categorical(["1", "1", "2", "2", "2"]),
        "date number": [0, 0, 0, 0, 0],
        "home": pd.Categorical(["A", "A", "a", "b", "a"]),
        "away": pd.Categorical(["B", "B", "b", "c", "d"]),
        "winner": ["a", "a", "d", "d", "d"],
    }
    test = Matches(pd.DataFrame(test_cols).set_index(["id", "date number"]))

    id_to_prob = pd.Series(  # only ids "1" and "2" exist in matches
        index=["0", "1", "2", "3", "4"],
        data=[
            {(3, 0): 0, (1, 1): 1, (0, 3): 0},
            {(3, 0): 0, (1, 1): 0, (0, 3): 1},
            {(3, 0): 0, (1, 1): 1, (0, 3): 0},
            {(3, 0): 0.5, (1, 1): 0.5, (0, 3): 0},
            {(3, 0): 0.33, (1, 1): 0.33, (0, 3): 0.34},
        ],
    )

    first_id_vars = [
        np.var([0, 6], ddof=1),
        np.var([0, 6], ddof=1),
        np.var([0, 6], ddof=1),
        np.var([0, 6], ddof=1),
        np.var([0, 6], ddof=1),
        np.var([0, 6], ddof=1),
    ]

    second_id_vars = [
        np.var([2, 2, 1, 1], ddof=1),
        np.var([2, 2, 1, 1], ddof=1),
        np.var([2, 2, 1, 1], ddof=1),
        np.var([2, 2, 1, 1], ddof=1),
        np.var([2, 2, 1, 1], ddof=1),
        np.var([2, 2, 1, 1], ddof=1),
    ]

    expected = pd.DataFrame(
        {
            "id": pd.Categorical(["1", "2"]),
            "real var": [np.var([0, 6], ddof=1), np.var([2, 2, 1, 1], ddof=1)],
            "mean": [mean(*first_id_vars), mean(*second_id_vars)],
            "0.950-quantile": [nfp(*first_id_vars), nfp(*second_id_vars)],
        }
    ).set_index("id")

    result = vst.get_kwargs_from_matches(
        test,
        num_iteration_simulation=(2, 3),
        winner_type="winner",
        winner_to_points={"h": (3, 0), "d": (1, 1), "a": (0, 3)},
        id_to_probabilities=id_to_prob,
    )
    assert result["df"].equals(expected)
    result = vst.get_kwargs_from_matches(
        test,
        num_iteration_simulation=(3, 2),
        winner_type="winner",
        winner_to_points={"h": (3, 0), "d": (1, 1), "a": (0, 3)},
        id_to_probabilities=id_to_prob,
    )
    assert result["df"].equals(expected)


def test_get_kwargs_from_matches_bigger_in_to_probabilities_new_points():
    test_cols = {
        "id": pd.Categorical(["1", "1", "2", "2", "2"]),
        "date number": [0, 0, 0, 0, 0],
        "home": pd.Categorical(["A", "A", "a", "b", "a"]),
        "away": pd.Categorical(["B", "B", "b", "c", "d"]),
        "winner": ["a", "a", "d", "d", "d"],
    }
    test = Matches(pd.DataFrame(test_cols).set_index(["id", "date number"]))

    id_to_prob = pd.Series(  # only ids "1" and "2" exist in matches
        index=["0", "1", "2", "3", "4"],
        data=[
            {(3, 0): 0, (1, 1): 1, (0, 3): 0},
            {(2, 1): 0, (0, 0): 0, (1, 2): 1},
            {(2, 1): 0, (0, 0): 1, (1, 2): 0},
            {(3, 0): 0.5, (1, 1): 0.5, (0, 3): 0},
            {(3, 0): 0.33, (1, 1): 0.33, (0, 3): 0.34},
        ],
    )

    first_id_vars = [
        np.var([2, 4], ddof=1),
        np.var([2, 4], ddof=1),
        np.var([2, 4], ddof=1),
        np.var([2, 4], ddof=1),
        np.var([2, 4], ddof=1),
        np.var([2, 4], ddof=1),
    ]

    second_id_vars = [
        np.var([0, 0, 0, 0], ddof=1),
        np.var([0, 0, 0, 0], ddof=1),
        np.var([0, 0, 0, 0], ddof=1),
        np.var([0, 0, 0, 0], ddof=1),
        np.var([0, 0, 0, 0], ddof=1),
        np.var([0, 0, 0, 0], ddof=1),
    ]

    expected = pd.DataFrame(
        {
            "id": pd.Categorical(["1", "2"]),
            "real var": [np.var([2, 4], ddof=1), np.var([0, 0, 0, 0], ddof=1)],
            "mean": [mean(*first_id_vars), mean(*second_id_vars)],
            "0.950-quantile": [nfp(*first_id_vars), nfp(*second_id_vars)],
        }
    ).set_index("id")

    result = vst.get_kwargs_from_matches(
        test,
        num_iteration_simulation=(2, 3),
        winner_type="winner",
        winner_to_points={"h": (2, 1), "d": (0, 0), "a": (1, 2)},
        id_to_probabilities=id_to_prob,
    )
    assert result["df"].equals(expected)
    result = vst.get_kwargs_from_matches(
        test,
        num_iteration_simulation=(3, 2),
        winner_type="winner",
        winner_to_points={"h": (2, 1), "d": (0, 0), "a": (1, 2)},
        id_to_probabilities=id_to_prob,
    )
    assert result["df"].equals(expected)
