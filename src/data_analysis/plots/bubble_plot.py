from typing import Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DESIRED_COLUMNS = [
    "proportion increase > 0",
    "avg increase [increase > 0]",
    "avg decrease [increase < 0]",
]
SPORT_TO_COLOR = {
    "handball": "green",
    "basketball": "blue",
    "soccer": "black",
    "volleyball": "red",
}
INDEX_KEYWORDS = {
    "O_Max_": ("maximizer_mirrored", "current"),
    "Y_Max_": ("maximizer_mirrored", "previous"),
    "O_Max_MinBreak": ("maximizer_break_min_mirrored", "current"),
    "Y_Max_MinBreak": ("maximizer_break_min_mirrored", "previous"),
    "O_MaxMin_": ("maximizer_reversed", "current"),
    "Y_MaxMin_": ("maximizer_reversed", "previous"),
    "O_MinBreak_MaxMin": ("maximizer_break_min_reversed", "current"),
    "Y_MinBreak_MaxMin": ("maximizer_break_min_reversed", "previous"),
}


def select_sport_all_tournaments(df: pd.DataFrame, sports: list[str]) -> pd.DataFrame:
    return df.loc(axis="index")[list(sports), "all"].reset_index(1, drop=True)


def select_from_keywords_in_index_level(
    df: pd.DataFrame,
    algorithm_type: Literal["graph", "recursive"],
    name_to_keywords: dict[str, tuple[str]] = INDEX_KEYWORDS,
    index_level: int = -1,
) -> np.ndarray:

    def _get_index_level() -> pd.Index:
        return df.index.get_level_values(index_level)

    def _find_entries_with_all_keywords() -> pd.DataFrame:
        _desired_entries = np.ones(len(index), dtype=bool)

        for keyword in keywords:
            _desired_entries &= index.str.contains(keyword)

        return df.loc[_desired_entries]

    def _format_filtered_df():
        filtered_df["name"] = name
        return (
            filtered_df.reset_index(index_level, drop=True)
            .rename_axis(["sport"], axis="index")
            .set_index("name", append=True)
        )

    all_names_df = []

    for name, keywords in name_to_keywords.items():

        index = _get_index_level()
        keywords = [algorithm_type] + list(keywords)
        filtered_df = _find_entries_with_all_keywords().copy()
        all_names_df.append(_format_filtered_df())

    return pd.concat(all_names_df).sort_index()


def plot_bubble_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    algorithm_type: Literal["recursive", "graph"],
    sport_to_balance_increase_df: dict[str, pd.DataFrame],
    sports: list[str],
    columns: str = DESIRED_COLUMNS,
):
    def _build_marker_from_name(_name: str):
        _oracle, _maxmin, _minbreaks = _name.split("_")
        return (
            r"$\mathrm{"
            + _oracle
            + "^{"
            + _maxmin
            + "}"
            + r"_{"
            + _minbreaks
            + r"}"
            + r"}$"
        )

    def _build_legend_handles():
        return [
            mpatches.Patch(color=color, label=sport.title(), alpha=0.5)
            for sport, color in SPORT_TO_COLOR.items()
        ]

    balance_increase_df = pd.concat(sport_to_balance_increase_df)
    df = select_sport_all_tournaments(balance_increase_df, sports)

    bubble_plot_df = select_from_keywords_in_index_level(df, algorithm_type)[columns]
    name_index = bubble_plot_df.index.get_level_values("name")
    sport_index = bubble_plot_df.index.get_level_values("sport")

    for sport in sports:
        for name in name_index.unique():

            filter_index = (sport_index == sport) & (name_index == name)
            filtered_df = bubble_plot_df[filter_index]

            params = {
                "x": filtered_df["avg increase [increase > 0]"],
                "y": filtered_df["proportion increase > 0"],
                "marker": _build_marker_from_name(name),
                "s": 5000,
                "alpha": 0.3,
                "label": sport.title(),
                "color": SPORT_TO_COLOR[sport],
            }
            ax.scatter(**params)

    ax.set_xlim(0.12, 0.36)
    ax.set_xlabel("Avg Increase")
    ax.set_ylim(0.4, 1)
    ax.set_ylabel("Proportion Increase")
    ax.legend(handles=_build_legend_handles(), loc=4)
