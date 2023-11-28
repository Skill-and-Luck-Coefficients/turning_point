from enum import Enum
from typing import Literal

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

# col names to desired name
HUE_COLUMNS: dict[tuple[str, str], str] = {
    ("%turning point", "normal"): r"Observed: $\tau_\%$",
    ("%turning point", "mean"): r"Expected: $\hat\tau_\%$",
}


class BoxplotDFCols(Enum):
    x = "x"
    value = "value"
    hue = "hue"


def _add_hue_column(
    df: pd.DataFrame, hue_columns: dict[str, str] | dict[tuple[str], str]
) -> pd.DataFrame:
    all_dfs: list[pd.DataFrame] = []

    for col_name, label in hue_columns.items():
        tp_df = pd.DataFrame()
        tp_df[BoxplotDFCols.value] = df[col_name]
        tp_df[BoxplotDFCols.hue] = label

        all_dfs.append(tp_df)

    return pd.concat(all_dfs)


def _create_boxplot_df(
    sport_to_tp_comparison: dict[str, pd.DataFrame],
    x_column: Literal["sport"] | None,
    hue_columns: dict[str, str] | dict[tuple[str], str],
) -> pd.DataFrame:
    turning_points_all_sports: list[pd.DataFrame] = []

    for sport, comparison in sport_to_tp_comparison.items():
        comparison = _add_hue_column(comparison, hue_columns)

        if x_column is not None:
            comparison[BoxplotDFCols.x] = sport.title()

        turning_points_all_sports.append(comparison)

    return pd.concat(turning_points_all_sports)


def plot_comparison_boxplot(
    ax: Axes,
    sport_to_df: dict[str, pd.DataFrame],
    x_column: Literal["sport"] | None = "sport",
    hue_columns: dict[str, str] | dict[tuple[str], str] = HUE_COLUMNS,
    legend: bool = True,
):
    """
    hue_columns:
        Maps turning points column name to the name it should be called in the plot.

        Notice that, since this is a comparasion, there should be two key/values pairs.
    """
    boxplot_df = _create_boxplot_df(
        sport_to_df,
        x_column,
        hue_columns,
    )

    # with sns.plotting_context(**BOXPLOT_THEME):
    sns.boxplot(
        data=boxplot_df,
        y=BoxplotDFCols.value,
        x=BoxplotDFCols.x if x_column is not None else None,
        hue=BoxplotDFCols.hue,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    if legend:
        ax.legend(title="")
    else:
        ax.legend().remove()
