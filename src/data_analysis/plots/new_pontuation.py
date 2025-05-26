import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .comparison_boxplot import plot_comparison_boxplot
from .comparison_scatter import plot_scatter_according_to_line_x_equals_y


def plot_new_pontuation_system(
    fig: Figure,
    axs: list[Axes],
    sport_to_tp_comparison: dict[str, pd.DataFrame],
    tp_columns: dict[str, str] | dict[tuple[str], str],
    no_difference_margin: float = 0.01,
    no_x_axis_data: bool = False,
):

    def _remove_x_axis_labels_if_necessary():
        for _ax in axs:
            if no_x_axis_data:
                _ax.set_xticklabels([])
                _ax.set_xlabel("")

    ax = axs[0]
    plot_scatter_according_to_line_x_equals_y(
        fig,
        [[ax]],
        sport_to_tp_comparison,
        tp_columns,
        no_difference_margin=no_difference_margin,
        # comparison kwargs
        pearson_corr_kwargs=None,
        y_gt_x_percent_kwargs=None,
        title_as_text_kwargs=None,
    )
    ax.set_title("")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_xticks([0.25, 0.5, 0.75])

    ax = axs[1]
    plot_comparison_boxplot(
        ax, sport_to_tp_comparison, hue_columns=tp_columns, legend=False
    )

    ax.set_title("")
    ax.set_yticks([0.25, 0.5, 0.75])
    xtick_lables = ["3-2 score\n3 winner\n0 loser", "3-2 score\n2 winner\n1 loser"]
    ax.set_xticks([-0.20, 0.20], xtick_lables)

    _remove_x_axis_labels_if_necessary()
