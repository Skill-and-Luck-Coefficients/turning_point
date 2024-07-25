from itertools import product

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


def get_key_to_balance_increase_all_sports(
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


def _replace_yesterday_values_with_only_div1(
    balance_increase: pd.DataFrame,
) -> pd.DataFrame:
    div1_increase = balance_increase.loc[["div1", "drr_div1"]].copy()
    previous_index = div1_increase.index.get_level_values(-1).str.contains("-previous")
    div1_previous_values = div1_increase.loc[previous_index].to_numpy()

    new_balance_increase = balance_increase.loc[["all", "drr"]].copy()
    new_balance_increase.loc[previous_index] = div1_previous_values
    return new_balance_increase


def get_sport_to_balance_increase_df(
    key_to_sport_to_balance_increase: dict[str, pd.DataFrame],
    sports: list[str],
):
    """
    Maps each sport (and 'total') to their balance increase dataframe.

    Remark: Yesterday model entries are replaced by the division 1 values.
    """
    balance_increase_df = pd.concat(
        {
            key: pd.concat(sport_to_balance_increase, axis="columns")
            for key, sport_to_balance_increase in key_to_sport_to_balance_increase.items()
        },
        axis="columns",
    ).transpose()

    balance_increase_df = _replace_yesterday_values_with_only_div1(balance_increase_df)

    sport_to_increase = {
        sport: balance_increase_df.loc(axis="index")[:, sport].reset_index(1, drop=True)
        for sport in sports
    }
    sport_to_increase["total"] = balance_increase_df.groupby(level=[0, 2]).mean()
    return sport_to_increase


def create_latex_table_rows(balance_increase_one_sport: pd.DataFrame) -> str:
    KEYS = {
        "index": list(
            product(
                ("drr", "all"),
                ("graph", "recursive"),
                ("current", "previous"),
                ("mirrored", "reversed"),
                ("random_", ""),
            )
        ),
        "checkmark": list(product((r"\cm", "   "), repeat=5)),
    }

    COLUMNS = [
        "proportion increase > 0",
        "avg increase [increase > 0]",
        "avg decrease [increase < 0]",
    ]

    def _build_tournament_index(
        format: str, algorithm: str, oracle: str, second_turn: str, best_home: str
    ) -> tuple[str, str]:
        return (format, f"{algorithm}_tp_maximizer_{best_home}{second_turn}-{oracle}")

    def _create_latex_table_row(table_values: str, checkmark_entries: tuple) -> str:
        checkmark_str = " & ".join(checkmark_entries)
        table_values_str = " & ".join(f"{number:5.1%}" for number in table_values)
        return checkmark_str + " & " + table_values_str

    rows = []

    for index_keys, checkmark_keys in zip(KEYS["index"], KEYS["checkmark"]):
        index = _build_tournament_index(*index_keys)

        if index not in balance_increase_one_sport.index:
            continue

        table_values = balance_increase_one_sport.loc[index, COLUMNS]
        row = _create_latex_table_row(table_values, checkmark_keys)
        rows.append(row + r" \\")

    return "\n".join(rows)
