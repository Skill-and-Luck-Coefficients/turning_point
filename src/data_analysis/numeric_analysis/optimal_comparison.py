import pandas as pd


def get_optimal_comparison(
    sport_to_comparison: dict[str, pd.DataFrame],
    optimal_columns: list[str],
    tp_column: str = "%turning point",
):
    """
    Get comparison between optimal columns and real/permuted columns.
    """
    all_cols = {}

    for optimal_col in optimal_columns:
        all_sports = []

        for sport, comparison in sport_to_comparison.items():
            tps = comparison[tp_column]

            one_sport = pd.Series(name=sport, dtype="float64")
            one_sport["< 2.5%"] = (tps[optimal_col] < tps["2.5%"]).mean()
            one_sport["< real"] = (tps[optimal_col] < tps["normal"]).mean()
            one_sport["> real"] = (tps[optimal_col] > tps["normal"]).mean()
            one_sport["> 97.5%"] = (tps[optimal_col] > tps["97.5%"]).mean()

            all_sports.append(one_sport)

        all_cols[optimal_col] = pd.concat(all_sports, axis="columns")

    optimal_comp = pd.concat(all_cols, axis="columns", names=["optimal col", "sport"])
    return optimal_comp.transpose()
