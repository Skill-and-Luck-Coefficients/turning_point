from dataclasses import dataclass

import pandas as pd
import pytest

import turning_point.permutation_coefficient as pc


@dataclass
class TurningPointMock:
    df: pd.DataFrame


@dataclass
class PermutationStatsMock:
    statistical_measures_mock: pd.DataFrame

    def statistical_measures(self, percentiles: list[float]) -> pd.DataFrame:
        return self.statistical_measures_mock


@dataclass
class PermutationOptimalMock:
    optimal_mock: pd.DataFrame

    def unstack_permutation_id(self) -> pd.DataFrame:
        return self.optimal_mock


@pytest.fixture
def normal():
    return TurningPointMock(
        pd.DataFrame(
            {
                "id": pd.Categorical(["1", "2"]),
                "turning point": [20, 35],
                "%turning point": [0.35, 0.89],
            }
        ).set_index("id")
    )


@pytest.fixture
def permutation():
    return PermutationStatsMock(
        pd.DataFrame(
            {
                ("id", ""): pd.Categorical(["1", "2"]),
                ("turning point", "mean"): [10, 20],
                ("turning point", "std"): [1, 1],
                ("turning point", "2.5%"): [1, 2],
                ("turning point", "50%"): [11, 21],
                ("turning point", "97.5%"): [20, 35],
                ("%turning point", "mean"): [0.5, 0.6],
                ("%turning point", "std"): [0.1, 0.15],
                ("%turning point", "2.5%"): [0.05, 0.12],
                ("%turning point", "50%"): [0.52, 0.61],
                ("%turning point", "97.5%"): [0.85, 0.97],
            }
        ).set_index("id")
    )


@pytest.fixture
def optimal():
    return PermutationOptimalMock(
        pd.DataFrame(
            {
                ("id", ""): pd.Categorical(["1", "2"]),
                ("turning point", "minimizer"): [10, 20],
                ("turning point", "maximizer"): [1, 1],
                ("%turning point", "minimizer"): [0.5, 0.6],
                ("%turning point", "maximizer"): [0.1, 0.15],
            }
        ).set_index("id")
    )


@pytest.fixture
def full_expected():
    return pd.DataFrame(
        {
            ("id", ""): pd.Categorical(["1", "2"]),
            ("turning point", "normal"): [20, 35],
            ("turning point", "mean"): [10, 20],
            ("turning point", "std"): [1, 1],
            ("turning point", "2.5%"): [1, 2],
            ("turning point", "50%"): [11, 21],
            ("turning point", "97.5%"): [20, 35],
            ("turning point", "minimizer"): [10, 20],
            ("turning point", "maximizer"): [1, 1],
            ("%turning point", "normal"): [0.35, 0.89],
            ("%turning point", "mean"): [0.5, 0.6],
            ("%turning point", "std"): [0.1, 0.15],
            ("%turning point", "2.5%"): [0.05, 0.12],
            ("%turning point", "50%"): [0.52, 0.61],
            ("%turning point", "97.5%"): [0.85, 0.97],
            ("%turning point", "minimizer"): [0.5, 0.6],
            ("%turning point", "maximizer"): [0.1, 0.15],
        }
    ).set_index("id")


def test_get_turning_point_comparison__only_normal(normal, full_expected):
    result = pc.get_turning_point_comparison(normal)
    expected = full_expected[
        [("turning point", "normal"), ("%turning point", "normal")]
    ]
    assert result.equals(expected)


def test_get_turning_point_comparison__normal_perm(normal, permutation, full_expected):

    p = [2.5, 50, 97.5]
    result = pc.get_turning_point_comparison(normal, permutation, percentiles=p)

    expected = full_expected[
        [
            ("turning point", "normal"),
            ("turning point", "mean"),
            ("turning point", "std"),
            ("turning point", "2.5%"),
            ("turning point", "50%"),
            ("turning point", "97.5%"),
            ("%turning point", "normal"),
            ("%turning point", "mean"),
            ("%turning point", "std"),
            ("%turning point", "2.5%"),
            ("%turning point", "50%"),
            ("%turning point", "97.5%"),
        ]
    ]
    assert result.equals(expected)


def test_get_turning_point_comparison__normal_optimal(normal, optimal, full_expected):
    p = [2.5, 50]
    result = pc.get_turning_point_comparison(normal, optimal_tp=optimal, percentiles=p)
    expected = full_expected[
        [
            ("turning point", "normal"),
            ("turning point", "minimizer"),
            ("turning point", "maximizer"),
            ("%turning point", "normal"),
            ("%turning point", "minimizer"),
            ("%turning point", "maximizer"),
        ]
    ]
    assert result.equals(expected)


def test_get_turning_point_comparison(normal, permutation, optimal, full_expected):

    percentiles = [2.5, 50, 97.5]
    result = pc.get_turning_point_comparison(normal, permutation, optimal, percentiles)

    assert result.equals(full_expected)
