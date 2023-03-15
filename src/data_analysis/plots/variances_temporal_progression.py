import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from logs import turning_logger
from turning_point.normal_coefficient import TurningPoint
from turning_point.variance_stats import ExpandingVarStats

from .utils import utils

COLOR_MARKER_LABEL = {
    "real var": ("red", "X", r"Real: $V_t$"),
    "mean": ("forestgreen", "o", r"Mean Simul: $\mu_t$"),
    "0.950-quantile": ("orange", "^", r"95% Simul: $q_t$"),
}

TEXT_PARAMETERS = {
    "text_size": 30,
    "text_y_pos": 60,
    "text_ha": "right",  # ha is short for horizontal alignment
}


def _plot_variance_progression_one_tourney(
    ax: Axes,
    variances: pd.DataFrame,
    color_marker_label_dict: utils.CMLDict,
) -> None:

    x: list[int] = variances.index.get_level_values("final date").to_list()

    for column, (color, marker, _) in color_marker_label_dict.items():

        y = variances[column]
        ax.scatter(x, y.to_list(), color=color, marker=marker, alpha=0.3)


def _plot_turning_point_line_one_tourney(ax: Axes, turning_point: float) -> None:

    if np.isinf(turning_point):
        print("Turning Point is infinity!")
        return

    bottom, top = ax.get_ylim()
    ymin = bottom * 0.95
    ymax = top * 0.95

    ax.vlines(
        turning_point, ymin, ymax, color="black", linestyles="dashed", alpha=0.65, lw=5
    )


def _plot_turning_point_text_one_tourney(
    ax: Axes, turning_point: float, text_ha: str, text_y_pos: float, text_size: int
) -> None:

    if np.isinf(turning_point):
        turning_logger.warning("Invalid turning points (np.inf of np.nan).")
        return

    text: str = f"$\\tau$={turning_point:.0f}"  # tau: greek letter for turning point

    displacement: float = 3 if (text_ha == "left") else -3
    text_x_pos: int = turning_point + displacement

    ax.text(text_x_pos, text_y_pos, text, ha=text_ha, size=text_size)


def plot_variances_temporal_progression(
    sport_to_tp: dict[str, TurningPoint],
    sport_to_var_stats: dict[str, ExpandingVarStats],
    nrows: int,
    ncols: int,
    figsize: tuple[float, float],
    names_and_ids: tuple[tuple[str, str]],
    last_date: int,
) -> None:

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    flat_axs = utils.flatten_axes(axs)

    for ax, (name, id_) in zip(flat_axs, names_and_ids):

        sport = utils.get_sport_name_from_id(id_)

        variances: pd.DataFrame = sport_to_var_stats[sport].df.loc[id_].iloc[:last_date]
        turning_point: int = sport_to_tp[sport].df.loc[id_, "turning point"]

        ax.set_title(name)

        _plot_variance_progression_one_tourney(ax, variances, COLOR_MARKER_LABEL)
        _plot_turning_point_line_one_tourney(ax, turning_point)
        _plot_turning_point_text_one_tourney(ax, turning_point, **TEXT_PARAMETERS)

    utils.add_xlabels_nth_row(fig, axs, "Date", n=-1)
    utils.add_ylabels_to_nth_col(fig, axs, "Variances", n=0)

    utils.add_legend_from_color_maker_labels(flat_axs[0], COLOR_MARKER_LABEL.values())

    fig.subplots_adjust(
        left=0.11, bottom=0.11, right=0.955, top=0.94, wspace=0.17, hspace=0.31
    )
