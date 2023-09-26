import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import data_analysis.plots.utils.plot_functions as pf

COLOR_MARKER = {
    -1: ("red", "o"),
    0: ("black", "s"),
    1: ("blue", "^"),
}

TP_COLUMNS: dict[tuple[str, str], str] = {
    ("%turning point", "normal"): "Normal",
    ("%turning point", "mean"): "Permuted",
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


def _plot_with_colors(ax: Axes, x: pd.Series, y: pd.Series, no_diff_margin: float):
    inside_margin = np.abs(x - y) <= no_diff_margin
    diff = np.where(inside_margin, 0, np.sign(x - y))
    for diff_value, (color, marker) in COLOR_MARKER.items():
        color_index = diff == diff_value
        ax.scatter(
            x=x[color_index],
            y=y[color_index],
            c=color,
            marker=marker,
            alpha=0.25,
            s=100,
        )


def plot_comparison_scatter(
    fig: Figure,
    axs: list[list[Axes]],
    sport_to_tp_comparison: dict[str, pd.DataFrame],
    tp_columns: dict[str, str] | dict[tuple[str], str] = TP_COLUMNS,
    no_difference_margin: float = 3,
    pearson_corr_kwargs: dict[str, float | str] | None = PEARSON_KWARGS,
    title_as_text_kwargs: dict[str, float | str] | None = TEXT_TITLE_KWARGS,
):
    flat_axs = pf.flatten_axes(axs)

    col_x, col_y = tp_columns.keys()

    for ax, sport in zip(flat_axs, sport_to_tp_comparison):
        full_x = sport_to_tp_comparison[sport][col_x]
        full_y = sport_to_tp_comparison[sport][col_y]

        with pd.option_context("mode.use_inf_as_na", True):
            x_not_na, y_not_na = full_x.dropna(), full_y.dropna()
            intersection = x_not_na.index.intersection(y_not_na.index)

        x, y = full_x.loc[intersection].copy(), full_y.loc[intersection].copy()

        _plot_with_colors(ax, x, y, no_difference_margin)
        pf.plot_x_equals_y_line(ax, x, y)

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
