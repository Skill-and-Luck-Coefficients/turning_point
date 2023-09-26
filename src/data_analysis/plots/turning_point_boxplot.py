import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from logs import turning_logger
from turning_point.match_coefficient import MatchTurningPoint
from turning_point.normal_coefficient import TurningPoint


def _log_invalid_values(tp: pd.DataFrame, not_na_tp: pd.DataFrame):
    all_index: set[str] = set(tp.index.get_level_values("id"))
    not_na_index: set[str] = set(not_na_tp.index.get_level_values("id"))

    turning_logger.warning(
        f"Invalid turning points (np.inf of np.nan):\n"
        f"{sorted(all_index - not_na_index)}"
    )


def _concatenate_turning_points_into_one_df(
    sport_to_tp: dict[str, TurningPoint] | dict[str, MatchTurningPoint],
    tp_column: str,
) -> pd.DataFrame:
    all_sports_tps = []

    for sport, tp in sport_to_tp.items():
        tp_df = tp.df[[tp_column]]

        with pd.option_context("mode.use_inf_as_na", True):
            not_na_tp = tp_df.dropna().copy()
            _log_invalid_values(tp_df, not_na_tp)

        not_na_tp["sport"] = sport.title()

        # "%seasons" is used as xlabel for boxplot
        percent_with_tp = len(not_na_tp) / len(tp_df)
        not_na_tp["%Seasons"] = f"{percent_with_tp:.1%}"

        all_sports_tps.append(not_na_tp)

    return pd.concat(all_sports_tps)


def plot_boxplot_turning_points(
    axs: list[Axes],
    sport_to_tp: dict[str, TurningPoint],
    sport_to_mtp: dict[str, MatchTurningPoint],
    mtp_tp_column_to_use: dict[str, str],
):
    mtp_col, tp_col = mtp_tp_column_to_use.keys()
    mtp_title, tp_title = mtp_tp_column_to_use.values()

    ax = axs[0]
    df = _concatenate_turning_points_into_one_df(sport_to_mtp, mtp_col)
    sns.boxplot(df, y=mtp_col, x="%Seasons", hue="sport", dodge=False, ax=ax)
    ax.set_ylabel("")
    ax.set_title(mtp_title.title())
    ax.legend(title="", fontsize=25)

    ax = axs[1]
    df = _concatenate_turning_points_into_one_df(sport_to_tp, tp_col)
    sns.boxplot(df, y=tp_col, x="%Seasons", dodge=False, ax=ax)
    ax.set_ylabel("")
    ax.set_title(tp_title.title())
