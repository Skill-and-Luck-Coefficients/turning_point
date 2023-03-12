import pandas as pd


def add_second_level_to_column_names(df: pd.DataFrame, name: str) -> pd.DataFrame:

    """
    Add second level to df columns index.

    Example:
        name : "ok"
        df: pd.DataFrame[
            index = ...

            columns = [
                "col1": ...
                "col2": ...
            ]
        ]

        Returns: pd.DataFrame[
            index = df.index

            columns = [
                ("col1", "ok"): df.col1
                ("col2", "ok"): df.col2
            ]
        ]
    """

    new_columns = pd.MultiIndex.from_product([df.columns, [name]])
    return df.set_axis(new_columns, axis="columns")
