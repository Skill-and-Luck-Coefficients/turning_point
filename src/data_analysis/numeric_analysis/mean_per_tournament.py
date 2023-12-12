import pandas as pd

NAME_COUNTRY_REGEX = r"(.+?)@/.+?/(.+?)/.+"
NAME_COUNTRY = ["name", "country"]


def get_mean_per_tournament(
    sport_to_comparison: dict[str, pd.DataFrame],
    desired_columns: list[tuple[str, str]],
    sort_by: tuple[str, str] = ("%turning point", "mean"),
    agg_func: str = "mean",
) -> pd.DataFrame:
    def _apply_agg_func(df: pd.DataFrame) -> pd.Series:
        """Withough this function, it does not consider NaN values correctly."""
        return df.agg(agg_func)

    all_sports: dict[str, pd.DataFrame] = {}

    for sport, comparison in sport_to_comparison.items():
        tps = comparison[desired_columns]

        name_country = tps.index.str.extract(NAME_COUNTRY_REGEX)
        new_index = pd.MultiIndex.from_frame(name_country, names=NAME_COUNTRY)

        with pd.option_context("mode.use_inf_as_na", True):
            mean_per_tournament = (
                tps.set_index(new_index).groupby(NAME_COUNTRY).apply(_apply_agg_func)
            )
        all_sports[sport] = mean_per_tournament.sort_values(sort_by)

    return pd.concat(all_sports, names=["sport"])
