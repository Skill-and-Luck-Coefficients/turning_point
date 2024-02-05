from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import data_analysis.plots.utils.plot_functions as pf

TP_COLUMNS: dict[tuple[str, str], str] = {
    ("%turning point", "normal"): r"Observed: $\tau_\%$",
    ("%turning point", "mean"): r"Expected: $\hat\tau_\%$",
}

CONFIDENCE_COLUMNS = [("%turning point", "2.5%"), ("%turning point", "97.5%")]

CONFIDENCE_PARAMETERS = {
    # COLOR, MARKER, SIZE, LABEL
    0: ("black", "*", 50, None),  # inside interval
    -1: ("red", "o", 100, r"Observed $\leq$ 2.5%"),  # below interval
    1: ("blue", "s", 100, r"Observed $\geq$ 97.5%"),  # above interval
}

LINE_X_EQUALS_Y_PARAMETERS = {
    # COLOR, MARKER, SIZE LABEL
    0: ("black", "s", 100, r"|x - y| $\leq$ "),  # close to x = y
    -1: ("red", "o", 100, " y - x  > "),  # above x = y
    1: ("blue", "*", 150, " x - y  > "),  # below x = y
}

BINNED_INTERVAL_PARAMETERS = {
    # COLOR, MARKER, SIZE, LABEL
    pd.Interval(-np.inf, np.inf, "both"): ("blue", "o", 85, r": 100%"),
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


def _remove_nan_values(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    with pd.option_context("mode.use_inf_as_na", True):
        x_not_na, y_not_na = x.dropna(), y.dropna()
        intersection = x_not_na.index.intersection(y_not_na.index)

    return x.loc[intersection].copy(), y.loc[intersection].copy()


def plot_comparison_scatter_all_sports_template(
    fig: Figure,
    axs: list[list[Axes]],
    sport_to_tp_comparison: dict[str, pd.DataFrame],
    plot_columns_to_label: dict[str, str] | dict[tuple[str], str],
    scatter_plot_func: Callable[[Axes, pd.Series, pd.Series, pd.DataFrame], None],
    x_equals_y_line: bool = True,
    linregress: bool = False,
    pearson_corr_kwargs: dict[str, float | str] | None = PEARSON_KWARGS,
    y_gt_x_percent_kwargs: dict[str, float | str] | None = PEARSON_KWARGS,
    title_as_text_kwargs: dict[str, float | str] | None = TEXT_TITLE_KWARGS,
):
    """
    `scatter_plot_func`: dict[str, str] | dict[tuple[str], str]
        First option (used for same plot for each sport):
            First entry: x-column
            Other entry: y-column (one value, same y-column for all)
        Second option (used for different plots for the same sport):
            First entry: x-column
            Other entries: y-columns (one value for each ax in axs)

            In this case, `sport_to_tp_comparison` should only be a pd.DataFrame
    """
    flat_axs = pf.flatten_axes(axs)

    col_x, *cols_y = list(plot_columns_to_label.keys())
    xlabel, *ylabels = list(plot_columns_to_label.values())

    if len(cols_y) == 1:
        cols_y = [cols_y[0] for _ in enumerate(flat_axs)]
    else:
        sport_to_tp_comparison = {col_y: sport_to_tp_comparison for col_y in cols_y}

    for ax, col_y, sport in zip(flat_axs, cols_y, sport_to_tp_comparison):
        full_x = sport_to_tp_comparison[sport][col_x]
        full_y = sport_to_tp_comparison[sport][col_y]

        x, y = _remove_nan_values(full_x, full_y)
        tp_comparison = sport_to_tp_comparison[sport].loc[x.index].copy()

        scatter_plot_func(ax, x, y, tp_comparison)

        if x_equals_y_line:
            pf.plot_x_equals_y_line(ax, x, y)

        if linregress:
            line_kws = {"color": "black", "alpha": 0.5}
            sns.regplot(ax=ax, x=x, y=y, scatter_kws={"alpha": 0}, line_kws=line_kws)
            ax.set_xlabel("")
            ax.set_ylabel("")

        if pearson_corr_kwargs is not None:
            corr_matrix = np.corrcoef(x, y)
            p_kwargs = PEARSON_KWARGS | pearson_corr_kwargs
            ax.text(s=f"Corr: {corr_matrix[0][1]:.2f}", **p_kwargs)

        if y_gt_x_percent_kwargs is not None:
            y_gt_x = (y > x).mean()
            ygx_kwargs = PEARSON_KWARGS | y_gt_x_percent_kwargs
            ax.text(s=f"y > x: {y_gt_x:.2%}", **ygx_kwargs)

        if title_as_text_kwargs is not None:
            t_kwargs = TEXT_TITLE_KWARGS | title_as_text_kwargs
            ax.text(s=str(sport).title(), **t_kwargs)
        else:
            ax.set_title(str(sport).title())

    pf.add_xlabels_nth_row(fig, axs, xlabel, n=-1)
    for ax, ylabel in zip(flat_axs, ylabels):
        ax.set_ylabel(ylabel)
        ax.set_title("")

    if len(ylabels) == 1:
        pf.add_ylabels_to_nth_col(fig, axs, ylabels[0], n=0)


def plot_scatter_according_to_line_x_equals_y(
    fig: Figure,
    axs: list[list[Axes]],
    sport_to_tp_comparison: dict[str, pd.DataFrame],
    plot_columns_to_label: dict[str, str] | dict[tuple[str], str],
    no_difference_margin: float = 0.01,
    scatter_parameters: dict[Literal[-1, 0, 1], Any] = LINE_X_EQUALS_Y_PARAMETERS,
    **comparison_scatter_kwargs: Any,
) -> None:
    """
    scatter_parameters: dict[int, tuple[Any]]
        Maps int to (color, marker, size, label).

        Three colors/markers:
            1:  x - y  >  no_difference_margin
            0: |x - y| <= no_difference_margin
            -1:  y - x  >  no_difference_margin

    **comparison_scatter_kwargs:
        See plot_comparison_scatter_all_sports_template parameters.
    """

    def _plot_scatter_according_to_line_x_equals_y(
        ax: Axes, x: pd.Series, y: pd.Series, tp_comparison: pd.DataFrame
    ):
        inside_margin = np.abs(x - y) <= no_difference_margin
        diff = np.where(inside_margin, 0, np.sign(x - y))

        for diff_value, (c, m, s, label) in scatter_parameters.items():
            label = f"{label}{no_difference_margin:.2f}"

            color_index = diff == diff_value
            color_x, color_y = x[color_index], y[color_index]

            ax.scatter(color_x, color_y, c=c, marker=m, label=label, alpha=0.25, s=s)

    plot_comparison_scatter_all_sports_template(
        fig,
        axs,
        sport_to_tp_comparison,
        plot_columns_to_label,
        scatter_plot_func=_plot_scatter_according_to_line_x_equals_y,
        **comparison_scatter_kwargs,
    )


def plot_scatter_according_to_confidence_interval(
    fig: Figure,
    axs: list[list[Axes]],
    sport_to_tp_comparison: dict[str, pd.DataFrame],
    plot_columns_to_label: dict[str, str] | dict[tuple[str], str],
    col_in_interval: Literal["x", "y"] | str | tuple[str] = "x",
    confidence_interval_columns: list[str] | list[tuple[str]] = CONFIDENCE_COLUMNS,
    scatter_parameters: dict[Literal[-1, 0, 1], tuple[Any]] = CONFIDENCE_PARAMETERS,
    **comparison_scatter_kwargs: Any,
) -> None:
    """
    col_in_interval: Literal["x", "y"] | str | tuple[str] = "x"
        Which values should be compared to the interval. Where "x" and "y" are the plot axis.

        If any string other than "x" or "y" is passed, it should be column.

    scatter_parameters: list[str] | list[tuple[str]]
        Which columns from tp_comparison should be used for the confidence interval.

        First entry: lower limit
        Second entry: upper limit

    scatter_parameters: dict[int, tuple[Any]]
        Maps int to (color, marker, size, label).

        Three colors/markers:
            -1: Real variance below confidence_interval_columns[0]
            0: Real variance inside confidence interval
            1: Real variance above confidence_interval_columns[1]

    **comparison_scatter_kwargs:
        See plot_comparison_scatter_all_sports_template parameters.
    """

    def _get_above_or_below(
        values: pd.Series, lower_limit: pd.Series, upper_limit: pd.Series
    ):
        below = -(values <= lower_limit).astype(int)  # below: -1
        above = (upper_limit <= values).astype(int)  # above: 1
        return below + above  # below: -1, inside: 0, above: 1

    def _plot_scatter_according_to_confidence_interval(
        ax: Axes, x: pd.Series, y: pd.Series, tp_comparison: pd.DataFrame
    ):
        lower_col, upper_col = confidence_interval_columns
        lower_limit, upper_limit = tp_comparison[lower_col], tp_comparison[upper_col]

        if col_in_interval in tp_comparison.columns:
            values = tp_comparison[col_in_interval]
        else:
            values = y if col_in_interval == "y" else x

        below_or_above = _get_above_or_below(values, lower_limit, upper_limit)

        for key, (c, m, s, label) in scatter_parameters.items():
            color_index = below_or_above == key
            color_x, color_y = x[color_index], y[color_index]
            ax.scatter(color_x, color_y, c=c, marker=m, label=label, alpha=0.25, s=s)

    plot_comparison_scatter_all_sports_template(
        fig,
        axs,
        sport_to_tp_comparison,
        plot_columns_to_label,
        scatter_plot_func=_plot_scatter_according_to_confidence_interval,
        **comparison_scatter_kwargs,
    )


def plot_scatter_according_to_binned_intervals(
    fig: Figure,
    axs: list[list[Axes]],
    sport_to_tp_comparison: dict[str, pd.DataFrame],
    plot_columns_to_label: dict[str, str] | dict[tuple[str], str],
    column_to_bin: dict[str | tuple[str], str] = {"%odds present": "%With Odds"},
    scatter_parameters: dict[pd.Interval, tuple[Any]] = BINNED_INTERVAL_PARAMETERS,
    **comparison_scatter_kwargs: Any,
) -> None:
    """
    binned_intervals_parameters: dict[pd.Interval, tuple[Any]]
        Maps interval to (color, marker, size, label).

    **comparison_scatter_kwargs:
        See plot_comparison_scatter_all_sports_template parameters.
    """

    def _plot_scatter_according_to_binned_intervals(
        ax: Axes, x: pd.Series, y: pd.Series, tp_comparison: pd.DataFrame
    ):
        (col_to_bin, col_label), *_ = column_to_bin.items()
        values_to_bin = tp_comparison[col_to_bin]

        for interval, (c, m, s, label) in scatter_parameters.items():
            label = f"{col_label}{label}"

            color_index = values_to_bin.apply(lambda value: value in interval)
            color_x, color_y = x[color_index], y[color_index]

            ax.scatter(color_x, color_y, c=c, marker=m, label=label, alpha=0.25, s=s)

    plot_comparison_scatter_all_sports_template(
        fig,
        axs,
        sport_to_tp_comparison,
        plot_columns_to_label,
        scatter_plot_func=_plot_scatter_according_to_binned_intervals,
        **comparison_scatter_kwargs,
    )
