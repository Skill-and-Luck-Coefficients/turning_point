from dataclasses import dataclass
from typing import Protocol, TypeVar

import pandas as pd

REGEX_COLUMNS = ["name", "sport", "country"]
NAME_SPORT_COUNTRY_REGEX = r"(.+?)@/(.+?)/(.+?)/.+"


class ContainDF(Protocol):
    df: pd.DataFrame


ContainDFType = TypeVar("ContainDFType", bound=ContainDF)


@dataclass
class Division:
    """
    Contains information about the division of all tournaments.

    df: pd.DataFrame[
        index = [
            "name": str
                Tournament name
            "sport": str
                Tournament sport
            "country": str
                Tournament country
        ]

        columns = [
            "division": int
                Tournament division
        ]
    ]
    """

    df: pd.DataFrame

    def __post_init__(self) -> None:
        index_in_cols = [
            index for index in REGEX_COLUMNS if index not in self.df.columns
        ]
        self.df = self.df.reset_index(index_in_cols).set_index(REGEX_COLUMNS)

    def filter_division_in_index(
        self,
        id_index: pd.Index | pd.MultiIndex | pd.Series | pd.DataFrame,
        division: int | list[int],
    ) -> list[str]:
        """
        Selects tournaments ids in `id_index` that are of the required `division`.

        ----
        Parameters:
            id_index: pd.Index | pd.MultiIndex | pd.Series | pd.DataFrame
                Tournament index.

                Must contain an index level named "id". Valid formats:
                    f"{tournament_name}@/{sport}/{country}/{current_name}-{year}"
                    f"{tournament_name}@/{sport}/{country}/{current_name}-{year}@{other_identifier}"

            division: int | list[int]
                Desired divisions

        -----
        Returns:
            list[str]
                List of tournaments ids corresponding to the desired divisions.
        """
        if isinstance(division, int):
            division = [division]

        if not isinstance(id_index, pd.Index):
            id_index = id_index.index
        id_index = id_index.get_level_values("id")

        extracted_names = id_index.str.extract(NAME_SPORT_COUNTRY_REGEX)
        name_sport_country = extracted_names.set_axis(REGEX_COLUMNS, axis="columns")
        to_expand_index = pd.MultiIndex.from_frame(name_sport_country)

        expanded_division = self.df.loc[to_expand_index]
        mask_index = expanded_division["division"].isin(set(division))
        return id_index[mask_index].unique().to_list()
