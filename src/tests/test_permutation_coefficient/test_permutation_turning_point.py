import numpy as np
import pandas as pd
import pytest

import turning_point.permutation_coefficient as pc


@pytest.fixture
def permutation_turning_point():
    return pc.PermutationTurningPoint(
        pd.DataFrame(
            {
                "id": ["2@0", "2@1", "1@ok@0", "1@ok@1", "1@ok@2", "1@ok@3"],
                "turning point": [10, 20, 30, 40, 25, 55],
                "%turning point": [0.5, 0.75, 0.2, 0.4, 0.6, 0.5],
            }
        ).set_index("id"),
    )


def test_unstack_permutation_id(permutation_turning_point: pc.PermutationTurningPoint):
    expected_tp = (
        pd.DataFrame(
            {
                "id": ["2", "1@ok"],
                "0": [10.0, 30],
                "1": [20.0, 40],
                "2": [np.nan, 25],
                "3": [np.nan, 55],
            }
        )
        .set_index("id")
        .sort_index()
        .rename_axis(["type"], axis=1)
    )

    result = permutation_turning_point.unstack_permutation_id()
    assert result["turning point"].equals(expected_tp)

    expected_percentp = (
        pd.DataFrame(
            {
                "id": ["2", "1@ok"],
                "0": [0.5, 0.2],
                "1": [0.75, 0.4],
                "2": [np.nan, 0.6],
                "3": [np.nan, 0.5],
            }
        )
        .set_index("id")
        .sort_index()
        .rename_axis(["type"], axis=1)
    )

    result = permutation_turning_point.unstack_permutation_id()
    assert result["%turning point"].equals(expected_percentp)


def test_statistical_measures(permutation_turning_point: pc.PermutationTurningPoint):
    expected_tp = (
        pd.DataFrame(
            {
                "id": pd.Categorical(["2", "1@ok"]),
                "mean": [np.mean([10, 20]), np.mean([30, 40, 25, 55])],
                "std": [np.std([10, 20], ddof=1), np.std([30, 40, 25, 55], ddof=1)],
                "2.5%": [
                    np.percentile([10, 20], 2.5),
                    np.percentile([30, 40, 25, 55], 2.5),
                ],
                "50%": [
                    np.percentile([10, 20], 50),
                    np.percentile([30, 40, 25, 55], 50),
                ],
                "97.5%": [
                    np.percentile([10, 20], 97.5),
                    np.percentile([30, 40, 25, 55], 97.5),
                ],
            }
        )
        .set_index("id")
        .sort_index()
    )

    result = permutation_turning_point.statistical_measures([2.5, 50, 97.5])
    assert result["turning point"].equals(expected_tp)

    expected_percentp = (
        pd.DataFrame(
            {
                "id": pd.Categorical(["2", "1@ok"]),
                "mean": [np.mean([0.5, 0.75]), np.mean([0.2, 0.4, 0.6, 0.5])],
                "std": [
                    np.std([0.5, 0.75], ddof=1),
                    np.std([0.2, 0.4, 0.6, 0.5], ddof=1),
                ],
                "2.5%": [
                    np.percentile([0.5, 0.75], 2.5),
                    np.percentile([0.2, 0.4, 0.6, 0.5], 2.5),
                ],
                "50%": [
                    np.percentile([0.5, 0.75], 50),
                    np.percentile([0.2, 0.4, 0.6, 0.5], 50),
                ],
                "97.5%": [
                    np.percentile([0.5, 0.75], 97.5),
                    np.percentile([0.2, 0.4, 0.6, 0.5], 97.5),
                ],
            }
        )
        .set_index("id")
        .sort_index()
    )

    assert result["%turning point"].equals(expected_percentp)
