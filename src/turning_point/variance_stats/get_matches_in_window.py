import numpy as np
import pandas as pd

from tournament_simulations.data_structures import Matches


def _get_mask_inside_interval(
    matches: Matches, first_date: int, last_date: int
) -> np.ndarray:  # np.ndarray[bool]
    date_number = matches.df.index.get_level_values("date number").to_numpy()
    return (first_date <= date_number) & (date_number <= last_date)


def _get_mask_tournaments_not_already_finished(
    matches: Matches, last_date: int
) -> np.ndarray:  # np.ndarray[bool]
    # apply func
    def _last_date_one_id(df: pd.DataFrame) -> int:
        return df.index.get_level_values("date number")[-1]

    id_to_last_date = matches.df.groupby("id", observed=True).apply(_last_date_one_id)

    ids = matches.df.index.get_level_values("id")
    has_already_finished = last_date <= id_to_last_date[ids]

    return has_already_finished.to_numpy()


def select_matches_inside_window(
    matches: Matches, first_date: int, last_date: int
) -> Matches:
    """
    Select only matches between first and last dates (both included).

    If last date is greater than the last tournament date,
    it will also not be selected. This is done to avoid repeating
    calculations already made before.

    ----
    Parameters:
        matches: pd.DataFrame[
            index=[
                "id" -> "{current_name}@/{sport}/{country}/{name-year}/",\n
                "date number" -> int (explained below),
            ],\n
            columns=[]
        ]
            Columns aren't really important for this, only index matter.

        first_date: int
            First date to be considered (included).

        last_date: int
            Last date to be considered (included).

    -----
    Returns:
        Copy of df_matches with all undesired matches/tournaments removed.
    """

    mask_interval = _get_mask_inside_interval(matches, first_date, last_date)
    mask_not_finished = _get_mask_tournaments_not_already_finished(matches, last_date)

    filtered_matches = matches.df[mask_interval & mask_not_finished]
    return Matches(filtered_matches)
