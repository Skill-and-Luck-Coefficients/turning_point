from itertools import zip_longest
from typing import Iterable, NewType, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

LEGEND_BACKGROUND_COLOR = "#EAEAF2"

ColorMarker = NewType("ColorMarker", tuple[str, str])
CMList = list[ColorMarker]

ColorMarkerLabel = NewType("ColorMarkerLabel", tuple[str, str, str])
CMLDict = dict[str, ColorMarkerLabel]
CMLList = list[ColorMarkerLabel]


def get_sport_name_from_id(tourney_id: str) -> str:
    return tourney_id.split("/")[1]


def flatten_axes(axs: list[list[Axes]]) -> list[Axes]:
    return [ax for row in axs for ax in row]


def transpose_axes(axs: list[list[Axes]]) -> list[list[Axes]]:
    return list(map(list, zip(*axs)))


def create_legend_marker(label: str, color: str, marker: str) -> Line2D:

    return Line2D(
        xdata=[],
        ydata=[],
        label=label,
        marker=marker,
        markerfacecolor=color,
        alpha=0.7,
        color=LEGEND_BACKGROUND_COLOR,
    )


def add_legend_from_color_maker_labels(
    ax: Axes, color_marker_label: Iterable[ColorMarkerLabel], **kwargs
) -> None:

    handles = [
        create_legend_marker(label, color, marker)
        for color, marker, label in color_marker_label
    ]

    ax.legend(handles=handles, **kwargs)


def add_title_to_nth_row(
    axs: list[list[Axes]], titles: Union[str, Iterable[str]], n: int, **kwargs
) -> None:

    if isinstance(titles, str):
        titles = [titles]

    first_row: list[Axes] = axs[n]

    for ax, title in zip_longest(first_row, titles, fillvalue=titles[-1]):
        ax.set_title(title, **kwargs)


def add_xlabels_nth_row(
    fig: Figure,
    axs: list[list[Axes]],
    xlabels: Union[str, Iterable[str]],
    n: int,
    **kwargs,
) -> None:

    if isinstance(xlabels, str):
        xlabels = [xlabels]

    last_row: list[Axes] = axs[n]

    for ax, xlabel in zip_longest(last_row, xlabels, fillvalue=xlabels[-1]):
        ax.set_xlabel(xlabel, **kwargs)

    fig.align_xlabels(last_row)


def add_ylabels_to_nth_col(
    fig: Figure,
    axs: list[list[Axes]],
    ylabels: Union[str, Iterable[str]],
    n: int,
    **kwargs,
) -> None:

    if isinstance(ylabels, str):
        ylabels = [ylabels]

    first_col: list[Axes] = [row[n] for row in axs]

    for ax, ylabel in zip_longest(first_col, ylabels, fillvalue=ylabels[-1]):
        ax.set_ylabel(ylabel, **kwargs)

    fig.align_ylabels(first_col)


def create_discrete_cmap_norm(
    cmap_name: str, norm_divisions: Iterable[float]
) -> tuple[LinearSegmentedColormap, BoundaryNorm]:

    """
    Two consecutive values of norm_division represent the interval where
    the colors will be mapped to.

    Example:
        colors = ["red", "green", "blue"]  # from a random cmap
        norm_divisions = (0, 4, 7, 11)
            "red"   -> interval [0, 4)
            "green" -> interval [4,7)
            "blue"  -> interval [7, infinity)
    """

    num_colors: int = len(norm_divisions) - 1

    discrete_cmap: LinearSegmentedColormap = plt.get_cmap(cmap_name, num_colors)

    norm = BoundaryNorm(norm_divisions, discrete_cmap.N)

    return discrete_cmap, norm


def create_discrete_cmap_norm_with_set_over(
    cmap_name: str, norm_divisions: Iterable[float], set_over_color: str
) -> tuple[LinearSegmentedColormap, BoundaryNorm]:

    """
    Two consecutive values of norm_division represent the interval where
    the colors will be mapped to.

    Example:
        colors = ["red", "green", "blue"]  # from a random cmap
        set_over_color = "grey"
        norm_divisions = (0, 4, 7, 11)
            "red"   -> interval [0, 4)
            "green" -> interval [4,7)
            "blue"  -> interval [7, 11)
            "grey"  -> interval [11, infinity)
    """

    discrete_cmap, norm = create_discrete_cmap_norm(cmap_name, norm_divisions)

    discrete_cmap.set_over(set_over_color)

    return discrete_cmap, norm


def get_center_x_value(ax: Axes) -> float:

    left, right = ax.get_xlim()

    return 0.5 * (left + right)
