from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from turning_point.variance_stats import ExpandingVarStats


def _calculate_boolean_sequences_size(sequence: Sequence) -> list[int]:
    counts = []
    same_in_a_row_count = 1
    for current, next in zip(sequence, sequence[1:]):
        if current == next:
            same_in_a_row_count += 1
        else:
            counts.append(same_in_a_row_count)
            same_in_a_row_count = 1
    counts.append(same_in_a_row_count)

    return counts


def _max_default_zero(list_: Iterable) -> float:
    return max(list_) if list_ else 0


def get_variance_fluctuating_close_to_envelope(
    sport_to_var_stats: dict[str, ExpandingVarStats],
    var_col: str = "real var",
    envelope_col: str = "0.950-quantile",
    max_sequence_size: int = 10,
) -> pd.DataFrame:
    """
    Counts in how many tournaments the real variance fluctuated around the envelope.

    Above Below Above: [Below interval size]
        Real variance escaped the envelope, fell down below it, and escaped again.

    Below Above Below: [Above interval size]
        Real variance was below the envelope, escaped it, and then fell down again.
    """
    all_sports: dict[str, pd.DataFrame] = {}

    for sport, var_stats in sport_to_var_stats.items():
        greater_than_envelope = var_stats.df[var_col] > var_stats.df[envelope_col]

        grouped_above_envelope = greater_than_envelope.groupby("id", observed=True)
        sequences = grouped_above_envelope.agg(_calculate_boolean_sequences_size)

        fluctuation_sizes = {
            "above_below_above": sequences.apply(lambda list_: list_[2::2]),
            "below_above_below": sequences.apply(lambda list_: list_[1:-1:2]),
        }
        fluctuation_max_sizes = {
            name: sizes.apply(_max_default_zero)
            for name, sizes in fluctuation_sizes.items()
        }

        all_sports[sport] = pd.concat(
            {
                name: pd.Series(
                    {n: np.mean(sizes >= n) for n in range(1, max_sequence_size)}
                )
                for name, sizes in fluctuation_max_sizes.items()
            },
            axis="columns",
        )

    return pd.concat(all_sports).unstack(0)
