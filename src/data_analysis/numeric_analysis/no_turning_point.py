import pandas as pd


def get_no_tp(
    sport_to_comparison: dict[str, pd.DataFrame],
    desired_columns: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    Get all tournaments without a valid turning point.
    """
    all_sports_no_tp: dict[str, pd.DataFrame] = {}

    for sport, comparison in sport_to_comparison.items():
        with pd.option_context("mode.use_inf_as_na", True):
            index_infinity = comparison[desired_columns].isna().any(axis="columns")

        all_sports_no_tp[sport] = comparison[index_infinity]

    return pd.concat(all_sports_no_tp, names=["sport"]).sort_index()
