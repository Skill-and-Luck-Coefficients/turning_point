import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from data_analysis.plots.utils import plot_functions as pf
from logs import turning_logger
from turning_point.normal_coefficient import TurningPoint
from turning_point.variance_stats import ExpandingVarStats

COLOR_MARKER_LABEL = {
    "real var": ("red", "X", r"Observed: $V_t$"),
    "mean": ("darkgreen", "o", r"Expected Simul: $\mu_t$"),
    # "0.950-quantile": ("orange", "^", r"95% Simul: $q_t$"),
}

FILL_BETWEEN = {
    "mean": "forestgreen",
    "0.950-quantile": "darkgreen",
}

TEXT_PARAMETERS = {
    "text_size": 30,
    "text_y_pos": 60,
    "text_ha": "right",  # ha: horizontal alignment
}


def _plot_variance_progression_one_tourney(
    ax: Axes,
    variances: pd.DataFrame,
    color_marker_label_dict: pf.CMLDict,
    lower_envelope: pd.DataFrame | None = None,
) -> None:
    x: list[int] = variances.index.get_level_values("final date").to_list()

    expected, envelope = FILL_BETWEEN.keys()
    color = FILL_BETWEEN[expected]

    lower_y, upper_y = variances[expected], variances[envelope]
    if lower_envelope is not None:
        lower_y = lower_envelope[envelope]
        ax.plot(x, lower_y, color=FILL_BETWEEN[envelope], alpha=0.25)

    ax.fill_between(x, lower_y, upper_y, color=color, alpha=0.1, lw=0)
    ax.plot(x, upper_y, color=FILL_BETWEEN[envelope], alpha=0.25)

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
        x=turning_point,
        ymin=ymin,
        ymax=ymax,
        color="black",
        linestyles="dashed",
        alpha=0.65,
        lw=5,
    )


def _plot_turning_point_text_one_tourney(
    ax: Axes, turning_point: float, text_ha: str, text_y_pos: float, text_size: int
) -> None:
    if np.isinf(turning_point):
        turning_logger.warning("Invalid turning points (np.inf of np.nan).")
        return

    text: str = f"$\\tau$={turning_point:.0f}"  # tau: greek letter for turning point

    displacement = 3 if (text_ha == "left") else -3
    text_x_pos = turning_point + displacement

    ax.text(text_x_pos, text_y_pos, text, ha=text_ha, size=text_size)


def plot_variances_temporal_progression(
    fig: Figure,
    axs: list[list[Axes]],
    sport_to_tp: dict[str, TurningPoint],
    sport_to_var_stats: dict[str, ExpandingVarStats],
    names_and_ids: tuple[tuple[str, str]],
    last_date: int,
    sport_to_lower_envelope_bound: dict[str, ExpandingVarStats] | None = None,
) -> None:
    flat_axs = pf.flatten_axes(axs)

    for ax, (name, id_) in zip(flat_axs, names_and_ids):
        sport = pf.get_sport_name_from_id(id_)

        variances = sport_to_var_stats[sport].df
        turning_point = sport_to_tp[sport].df

        filtered_var: pd.DataFrame = variances.loc[id_].iloc[:last_date]
        filtered_tp: int = turning_point.loc[id_, "turning point"]

        filtered_lower_var = None
        if sport_to_lower_envelope_bound is not None:
            lower_variances = sport_to_lower_envelope_bound[sport].df
            filtered_lower_var: pd.DataFrame = lower_variances.loc[id_].iloc[:last_date]

        ax.set_title(name)

        _plot_variance_progression_one_tourney(
            ax, filtered_var, COLOR_MARKER_LABEL, filtered_lower_var
        )
        _plot_turning_point_line_one_tourney(ax, filtered_tp)
        _plot_turning_point_text_one_tourney(ax, filtered_tp, **TEXT_PARAMETERS)

    pf.add_xlabels_nth_row(fig, axs, "Matchday", n=-1)
    pf.add_ylabels_to_nth_col(fig, axs, "Competitive Imbalance", n=0)

    pf.add_legend_from_color_maker_labels(flat_axs[0], COLOR_MARKER_LABEL.values())
