import pandas as pd


def _get_statistical_information(
    sport_to_df: dict[str, pd.DataFrame],
    desired_columns: list[tuple[str, str]],
) -> pd.DataFrame:
    all_descriptions: dict[str, pd.DataFrame] = {}

    for sport, comparison in sport_to_df.items():
        description_one_sport: dict[tuple[str, str], pd.DataFrame] = {}

        for col in desired_columns:  # different cols can have different pd.NA
            description_one_sport[col] = comparison[col].dropna().describe()

        names = ["tp_type", "tp_data"]
        all_descriptions[sport] = pd.concat(description_one_sport, names=names)

    return pd.concat(all_descriptions, axis="columns")


def get_statistical_information(
    sport_to_df: dict[str, pd.DataFrame],
    desired_columns: list[tuple[str, str]],
    inf_as_na: bool = True,
) -> pd.DataFrame:
    if not inf_as_na:
        return _get_statistical_information(sport_to_df, desired_columns)

    with pd.option_context("mode.use_inf_as_na", True):
        return _get_statistical_information(sport_to_df, desired_columns)
