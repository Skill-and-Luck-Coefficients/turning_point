import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from turning_point.permutation_coefficient import TurningPointComparison

BOXPLOT_THEME = {
    "context": "talk",
    "font_scale": 1.05,
    "rc": {
        "figure.figsize": (14, 8),
        "legend.fontsize": 12,
        "xtick.labelsize": 14,
    },
}


def _create_boxplot_df(
    sport_to_tp_comparison: dict[str, TurningPointComparison], column: str
) -> pd.DataFrame:

    TITLES = ["Normal", "Permuted"]
    SECOND_LEVEL_COLUMNS = ["normal", "mean"]

    turning_points_all_sports: list[pd.DataFrame] = []

    for sport, comparison in sport_to_tp_comparison.items():

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            col_comparison: pd.DataFrame = comparison.comparison[column]

        comparison = col_comparison[SECOND_LEVEL_COLUMNS].set_axis(
            TITLES, axis="columns"
        )

        for title in TITLES:

            tp_df: pd.DataFrame = comparison[[title]].copy()
            tp_df[column] = comparison[title]
            tp_df["Normal/Permuted"] = title
            tp_df["Sport"] = sport.title()

            turning_points_all_sports.append(tp_df)

    return pd.concat(turning_points_all_sports).drop(TITLES, axis="columns")


def plot_comparison_boxplot(
    sport_to_tp_comparison: dict[str, TurningPointComparison],
    column: str = "%turning point",
) -> None:

    boxplot_df = _create_boxplot_df(sport_to_tp_comparison, column)

    with sns.plotting_context(**BOXPLOT_THEME):

        _, ax = plt.subplots()

        sns.boxplot(
            data=boxplot_df, y="%turning point", x="Sport", hue="Normal/Permuted", ax=ax
        )
        ax.set_title("%Date")
        ax.set_ylabel("")
        ax.legend(title="")
