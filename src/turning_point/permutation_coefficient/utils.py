import numpy as np
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


def get_permutation_identifiers(df: pd.DataFrame) -> list[str]:
    """
    Given a dataframe, returns a list of permutation idenfiers (as strings).
    If there are no idenfiers, returns a list with an empty string.

    ----
    Parameters:
        df: pd.DataFrame
            DataFrame with "id" index level containing strings like:
                "{current_name}@/{sport}/{country}/{name-year}/"

                "{current_name}@/{sport}/{country}/{name-year}/@{identifier}"

    -----
    Returns:
        list[str]:
            List of permutation identifiers.
    """

    unique_ids: pd.Index = df.index.get_level_values("id").unique()

    pattern = r".+@.+@(.+)"
    all_permutation_numbers: pd.DataFrame = unique_ids.str.extract(pattern).fillna("")

    return list(np.unique(all_permutation_numbers.to_numpy().flatten()))


def get_data_with_identifier(
    df: pd.DataFrame, permutation_idenfier: str
) -> pd.DataFrame:
    """
    Get data corresponding to `permutation_id` from a dataframe.
    If permutation is an empty str, the entire dataframe will be returned instead.

    ----
    Parameters:
        df: pd.DataFrame
            DataFrame with "id" index level containing strings like:
                "{current_name}@/{sport}/{country}/{name-year}/"

                "{current_name}@/{sport}/{country}/{name-year}/@{identifier}"

        permutation_idenfier: str
            Permutation identifier

    -----
    Returns:
        pd.DataFrame:
            Filtered dataframe, that is, all entries with the desired identifier.
    """
    if permutation_idenfier == "":
        return df

    pattern = rf".+@.+@{permutation_idenfier}"
    filtered_index = df.index.get_level_values("id").str.fullmatch(pattern)

    return df[filtered_index]
