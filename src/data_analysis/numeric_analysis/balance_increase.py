import numpy as np
import pandas as pd


def _get_balance_increase_stats(
    df: pd.DataFrame,
    optimal_tp_columns: list[str],
    tp_column: str = "%turning point",
) -> pd.DataFrame:
    tp_df = df[tp_column]

    if tp_column == "%turning point":
        tp_df = tp_df.replace({np.inf: 1})

    increase = tp_df[optimal_tp_columns] - tp_df["normal"].to_frame().to_numpy()

    return pd.DataFrame(
        {
            "size": len(increase),
            "proportion increase > 0": (increase > 0).mean(),
            "proportion increase < 0": (increase < 0).mean(),
            "median increase": (increase.median()),
            "avg increase": (increase.mean()),
            "median increase [increase > 0]": (increase[increase > 0].median()),
            "avg increase [increase > 0]": (increase[increase > 0].mean()),
            "median decrease [increase < 0]": (increase[increase < 0].abs().median()),
            "avg decrease [increase < 0]": (increase[increase < 0].abs().mean()),
        }
    ).transpose()


def get_balance_increase_all_sports(
    sport_to_tp_df: dict[str, pd.DataFrame],
    sports: list[str],
    optimal_cols: list[str],
    tp_col: str = "%turning point",
) -> dict[str, pd.DataFrame]:
    kwargs = {"optimal_tp_columns": optimal_cols, "tp_column": tp_col}
    return {
        sport: _get_balance_increase_stats(sport_to_tp_df[sport], **kwargs)
        for sport in sports
    }
