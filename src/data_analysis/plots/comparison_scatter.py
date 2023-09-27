import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import data_analysis.plots.utils.plot_functions as pf

TP_COLUMNS: dict[tuple[str, str], str] = {
    ("%turning point", "normal"): "Real",
    ("%turning point", "mean"): "Permutation (mean)",
}

INTERVAL_COLOR_MARKER_LABELS = {
    0: ("black", "*", None),
    -1: ("red", "o", r"Real $\leq$ 2.5%"),
    1: ("darkblue", "s", r"Real $\geq$ 97.5%"),
}

ABOVE_BELOW_COLOR_MARKER_LABELS = {
    0: ("black", "s", r"|x - y| $\leq$ "),
    -1: ("red", "o", " y - x  > "),
    1: ("darkblue", "*", " x - y  > "),
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


def _get_tp_column(tp_columns: dict[str, str] | dict[tuple[str], str]) -> str:
    keys = list(tp_columns.keys())

    if keys and isinstance(keys[0], tuple):
        keys = [key for tuple_ in keys for key in tuple_]

    return keys[0]


def _plot_colors_above_below_x_equals_x(
    ax: Axes, x: pd.Series, y: pd.Series, no_diff_margin: float
):
    inside_margin = np.abs(x - y) <= no_diff_margin
    diff = np.where(inside_margin, 0, np.sign(x - y))
    for diff_value, (color, marker, label) in ABOVE_BELOW_COLOR_MARKER_LABELS.items():
        color_index = diff == diff_value
        ax.scatter(
            x=x[color_index],
            y=y[color_index],
            c=color,
            marker=marker,
            label=f"{label}{no_diff_margin:.2f}",
            alpha=0.25,
            s=100,
        )


def _plot_colors_outside_interval(
    ax: Axes, x: pd.Series, y: pd.Series, lower_limit: pd.Series, upper_limit: pd.Series
):
    lower_limit, upper_limit = lower_limit.loc[x.index], upper_limit.loc[x.index]

    below = -(x <= lower_limit).astype(int)  # below: -1
    above = (upper_limit <= x).astype(int)  # above: 1
    below_or_above = below + above  # below: -1, inside: 0, above: 1

    for key, (color, marker, label) in INTERVAL_COLOR_MARKER_LABELS.items():
        color_index = below_or_above == key
        ax.scatter(
            x=x[color_index],
            y=y[color_index],
            c=color,
            marker=marker,
            label=label,
            alpha=0.25,
            s=100,
        )


def plot_comparison_scatter(
    fig: Figure,
    axs: list[list[Axes]],
    sport_to_tp_comparison: dict[str, pd.DataFrame],
    tp_columns: dict[str, str] | dict[tuple[str], str] = TP_COLUMNS,
    no_difference_margin: float | None = 3,
    pearson_corr_kwargs: dict[str, float | str] | None = PEARSON_KWARGS,
    title_as_text_kwargs: dict[str, float | str] | None = TEXT_TITLE_KWARGS,
    x_equals_y_line: bool = True,
):
    """
    If no_difference_margin is None, it will color according with whether or not
    the values are inside the confidence interval [2.5%, 97.5%].
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

        if no_difference_margin is not None:
            _plot_colors_above_below_x_equals_x(ax, x, y, no_difference_margin)
        else:
            tp_column = _get_tp_column(tp_columns)
            lower_limit = sport_to_tp_comparison[sport][(tp_column, "2.5%")]
            upper_limit = sport_to_tp_comparison[sport][(tp_column, "97.5%")]
            _plot_colors_outside_interval(ax, x, y, lower_limit, upper_limit)

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

    fig.subplots_adjust(
        left=0.11, bottom=0.11, right=0.955, top=0.94, wspace=0.1, hspace=0.14
    )
