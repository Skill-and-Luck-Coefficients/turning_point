import pandas as pd
import seaborn as sns

from logs import turning_logger
from turning_point.match_coefficient import MatchTurningPoint
from turning_point.normal_coefficient import TurningPoint

COLUMN_TO_TYPE = {
    "turning point": "Dates",
    "%turning point": "%Dates",
    "match turning point": "Matches",
    "%match turning point": "%Matches",
}

BOXPLOT_THEME = {
    "context": "talk",
    "font_scale": 1.05,
    "rc": {"figure.figsize": (14, 6), "legend.fontsize": 15.6},
}


def _log_invalid_values(tp: pd.DataFrame, not_na_tp: pd.DataFrame):

    all_index: set[str] = set(tp.index.get_level_values("id"))
    not_na_index: set[str] = set(not_na_tp.index.get_level_values("id"))

    turning_logger.warning(
        f"Invalid turning points (np.inf of np.nan):\n"
        f"{sorted(all_index - not_na_index)}"
    )


def _concatenate_turning_points_into_one_df(
    sport_to_tp: dict[str, TurningPoint],
    sport_to_mtp: dict[str, MatchTurningPoint],
    mtp_tp_column_to_use: tuple[str, str],
    column_to_type: dict[str, str],
) -> pd.DataFrame:

    mtp_col, tp_col = mtp_tp_column_to_use

    all_sports_tps = []

    for sport in sport_to_tp:

        mtp_tp_copies = (
            sport_to_mtp[sport].df.rename(columns={mtp_col: tp_col}),
            sport_to_tp[sport].df,
        )

        for tp, tp_column in zip(mtp_tp_copies, mtp_tp_column_to_use):

            with pd.option_context("mode.use_inf_as_na", True):

                not_na_tp = tp.dropna().copy()
                _log_invalid_values(tp, not_na_tp)

            not_na_tp["sport"] = sport.title()

            # "data type" is used to divide seaborn box-plot into two axes
            not_na_tp["data type"] = column_to_type[tp_column]

            # "%seasons" is used as xlabel for boxplot
            percent_with_tp = len(not_na_tp) / len(tp)
            not_na_tp["%Seasons"] = f"{percent_with_tp:.1%}"

            all_sports_tps.append(not_na_tp)

    return pd.concat(all_sports_tps)


def plot_boxplot_turning_points(
    sport_to_tp: dict[str, TurningPoint],
    sport_to_mtp: dict[str, MatchTurningPoint],
    mtp_tp_column_to_use: tuple[str, str],
) -> None:

    _, tp_col = mtp_tp_column_to_use

    turning_points_boxplot = _concatenate_turning_points_into_one_df(
        sport_to_tp, sport_to_mtp, mtp_tp_column_to_use, COLUMN_TO_TYPE
    )

    with sns.plotting_context(**BOXPLOT_THEME):

        axs: sns.FacetGrid = sns.catplot(
            data=turning_points_boxplot,
            x="%Seasons",
            y=tp_col,
            col="data type",
            hue="sport",
            kind="box",
            sharey=False,
            dodge=False,
            legend_out=False,
        )

        # axs.set_xlabels(labelpad=13)
        axs.set_ylabels("")
        axs.set_titles("{col_name}")

        axs.add_legend(title="")
