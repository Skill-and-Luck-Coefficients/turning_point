from typing import Iterable

import pandas as pd


def agg_tuple_per_id(to_agg: Iterable[pd.Series]) -> pd.Series:
    """
    For each index, aggregate all of its corresponding
    pd.Series values into a tuple

    ----
    Returns:
        pd.Series[
            index = [
                <id>: Any
                    identifier value
            ]

            value = [
                "tuple": tuple
                    Aggregated tuple.

                    Value for a given "id":
                        tuple[
                            < value from to_agg[0][< id >],
                            < value from to_agg[1][< id >],
                            ...
                        ]
            ]
        ]
    """
    aggregated_df = pd.concat(to_agg, axis="columns")
    return aggregated_df.dropna().agg(tuple, axis="columns").rename("tuple")
