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

    assert vst._calculate_mean_quantile(test).equals(expected)


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

    assert vst._calculate_mean_quantile(test).equals(expected)


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

    assert vst.get_kwargs_from_variances(test)["df"].equals(expected)


def get_kwargs_from_matches():

    test_cols = {
        "id": pd.Categorical(["1", "1", "2", "2", "2"]),
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

    assert vst.get_kwargs_from_matches(test, (2, 3))["df"].equals(expected)
    assert vst.get_kwargs_from_matches(test, (3, 2))["df"].equals(expected)
