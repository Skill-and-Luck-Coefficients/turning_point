import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

import data_analysis.plots.utils.utils as utils
from turning_point.permutation_coefficient import TurningPointComparison


def plot_comparison_scatter(
    sport_to_tp_comparison: dict[str, TurningPointComparison],
    nrows: int,
    ncols: int,
    figsize: tuple[float, float],
    column: str = "%turning point",
) -> None:

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex="all", sharey="all")
    flat_axs: list[Axes] = [ax for row in axs for ax in row]

    for ax, sport in zip(flat_axs, sport_to_tp_comparison):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tp_column = sport_to_tp_comparison[sport].comparison[column]

        with pd.option_context("mode.use_inf_as_na", True):
            not_na = tp_column.dropna()

        ax.scatter(not_na["normal"], not_na["mean"], alpha=0.3)
        ax.set_title(sport.title())

        corr_matrix = np.corrcoef(not_na["normal"], not_na["mean"])
        ax.text(
            1, 0.2, f"{corr_matrix[0][1]:.2f}", ha="right", va="bottom", fontsize=30
        )

    utils.add_xlabels_nth_row(fig, axs, "Normal", n=-1)
    utils.add_ylabels_to_nth_col(fig, axs, "Permuted", n=0)

    fig.subplots_adjust(
        left=0.11, bottom=0.11, right=0.955, top=0.94, wspace=0.08, hspace=0.15
    )
