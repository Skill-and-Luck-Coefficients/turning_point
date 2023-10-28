from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import data_analysis.plots.utils.plot_functions as pf

TP_COLUMNS: dict[tuple[str, str], str] = {
    ("%turning point", "normal"): "Real",
    ("%turning point", "mean"): "Permutation (mean)",
}

INTERVAL_PLOT_PARAMERS = {  # COLOR, MARKER, SIZE, LABEL
    0: ("black", "*", 50, None),
    -1: ("red", "o", 100, r"Real $\leq$ 2.5%"),
    1: ("blue", "s", 100, r"Real $\geq$ 97.5%"),
}

ABOVE_BELOW_PLOT_PARAMERS = {  # COLOR, MARKER, SIZE LABEL
    0: ("black", "s", 100, r"|x - y| $\leq$ "),
    -1: ("red", "o", 100, " y - x  > "),
    1: ("blue", "*", 150, " x - y  > "),
}

PEARSON_KWARGS = {
    "x": 1,
    "y": 0.05,
    "ha": "right",
    "va": "bottom",
    "fontsize": 30,
}
TEXT_TITLE_KWARGS = {
    "x": 1,
    "y": 0.15,
    "ha": "right",
    "va": "bottom",
    "fontsize": 35,
}


def _get_tp_type(tp_columns: dict[str, str] | dict[tuple[str], str]) -> str:
    keys = list(tp_columns.keys())

    if keys and isinstance(keys[0], tuple):
        keys = [key for tuple_ in keys for key in tuple_]

    return keys[0]


def _plot_colors_above_below_x_equals_y(
    ax: Axes, x: pd.Series, y: pd.Series, no_diff_margin: float
):
    inside_margin = np.abs(x - y) <= no_diff_margin
    diff = np.where(inside_margin, 0, np.sign(x - y))
    for diff_value, (color, marker, size, label) in ABOVE_BELOW_PLOT_PARAMERS.items():
        color_index = diff == diff_value
        ax.scatter(
            x=x[color_index],
            y=y[color_index],
            c=color,
            marker=marker,
            label=f"{label}{no_diff_margin:.2f}",
            alpha=0.25,
            s=size,
        )


def _plot_colors_outside_interval(
    ax: Axes, x: pd.Series, y: pd.Series, lower_limit: pd.Series, upper_limit: pd.Series
):
    lower_limit, upper_limit = lower_limit.loc[x.index], upper_limit.loc[x.index]

    below = -(x <= lower_limit).astype(int)  # below: -1
    above = (upper_limit <= x).astype(int)  # above: 1
    below_or_above = below + above  # below: -1, inside: 0, above: 1

    for key, (color, marker, size, label) in INTERVAL_PLOT_PARAMERS.items():
        color_index = below_or_above == key
        ax.scatter(
            x=x[color_index],
            y=y[color_index],
            c=color,
            marker=marker,
            label=label,
            alpha=0.25,
            s=size,
        )


def _scatter_plot(
    ax: Axes,
    x: pd.Series,
    y: pd.Series,
    plot_type: Literal["interval", "outliers", "linregress"],
    **type_kwargs: Any,
) -> None:
    match plot_type:
        case "interval":
            _plot_colors_above_below_x_equals_y(ax, x, y, type_kwargs["margin"])

        case "outliers":
            tp_comparison: pd.DataFrame = type_kwargs["tp_comparison"]
            tp_type: str = type_kwargs["tp_type"]
            lower_limit = tp_comparison[(tp_type, "2.5%")]
            upper_limit = tp_comparison[(tp_type, "97.5%")]
            _plot_colors_outside_interval(ax, x, y, lower_limit, upper_limit)

        case "linregress":
            sns.regplot(
                x=x,
                y=y,
                scatter_kws={"color": "blue", "alpha": 0.25, "s": 40},
                line_kws={"color": "black", "alpha": 0.5},
                ax=ax,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")


def plot_comparison_scatter(
    fig: Figure,
    axs: list[list[Axes]],
    sport_to_tp_comparison: dict[str, pd.DataFrame],
    tp_columns: dict[str, str] | dict[tuple[str], str] = TP_COLUMNS,
    plot_type: Literal["interval", "outliers", "linregress"] = "interval",
    x_equals_y_line: bool = True,
    pearson_corr_kwargs: dict[str, float | str] | None = PEARSON_KWARGS,
    title_as_text_kwargs: dict[str, float | str] | None = TEXT_TITLE_KWARGS,
    plot_type_kwargs: dict[str, Any] = {"margin": 0.01},
):
    """
    plot_type:
        "interval":
            plot_type_kwargs (default) = {"margin": 0.01}

            Three colors/markers:
                1:  x - y  >  margin
                2: |x - y| <= margin
                3:  y - x  >  margin

        "outliers":
            Three colors/markers:
                1: Real variance below 0.025-quantile
                2: Real variance inside confidence interval
                3: Real variance above 0.975-quantile

        "linregress":
            Plots the linear regression between x and y.
    """
    flat_axs = pf.flatten_axes(axs)

    col_x, col_y = tp_columns.keys()

    for ax, sport in zip(flat_axs, sport_to_tp_comparison):
        full_x = sport_to_tp_comparison[sport][col_x]
        full_y = sport_to_tp_comparison[sport][col_y]

        with pd.option_context("mode.use_inf_as_na", True):
            x_not_na, y_not_na = full_x.dropna(), full_y.dropna()
            intersection = x_not_na.index.intersection(y_not_na.index)

        x, y = full_x.loc[intersection].copy(), full_y.loc[intersection].copy()

        if x_equals_y_line:
            pf.plot_x_equals_y_line(ax, x, y)

        plot_type_kwargs["tp_type"] = _get_tp_type(tp_columns)
        plot_type_kwargs["tp_comparison"] = sport_to_tp_comparison[sport]
        _scatter_plot(ax, x, y, plot_type, **plot_type_kwargs)

        if pearson_corr_kwargs is not None:
            corr_matrix = np.corrcoef(x, y)
            p_kwargs = PEARSON_KWARGS | pearson_corr_kwargs
            ax.text(s=f"Corr: {corr_matrix[0][1]:.2f}", **p_kwargs)

        if title_as_text_kwargs is not None:
            t_kwargs = TEXT_TITLE_KWARGS | title_as_text_kwargs
            ax.text(s=sport.title(), **t_kwargs)
        else:
            ax.set_title(sport.title())

    label_x, label_y = tp_columns.values()
    pf.add_xlabels_nth_row(fig, axs, label_x, n=-1)
    pf.add_ylabels_to_nth_col(fig, axs, label_y, n=0)
