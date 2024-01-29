import pandas as pd

from turning_point.metric_stats import ExpandingMetricStats
from turning_point.normal_coefficient import TurningPoint


def get_percentage_outside_envelope_before_tp(
    sport_to_var_stats: dict[str, ExpandingMetricStats],
    sport_to_tp: dict[str, TurningPoint],
) -> dict[str, pd.DataFrame]:
    all_sports: dict[str, pd.DataFrame] = {}

    for sport in sport_to_var_stats.keys():
        stats = sport_to_var_stats[sport].df
        tp = sport_to_tp[sport].df

        outside_envelope = stats["real"] > stats["quantile"]

        all_id_index = outside_envelope.index.get_level_values("id")
        expanded_tp = tp["turning point"].loc[all_id_index]
        before_tp = outside_envelope.index.get_level_values("final date") < expanded_tp

        # grouped_by_id = (outside_envelope & before_tp).groupby("id")
        grouped_by_id = outside_envelope.loc[before_tp].groupby("id")
        percent_outside = grouped_by_id.mean().rename("%outside")
        concated = pd.concat([percent_outside, tp["%turning point"]], axis="columns")

        all_sports[sport] = concated

    return all_sports
