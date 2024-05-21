from pathlib import Path

import pandas as pd

import turning_point.metric_stats as ms
import turning_point.normal_coefficient as nc
from logs import log, turning_logger

from .. import types
from . import utils


@log(turning_logger.info)
def _calculate_turning_point(filepath: Path) -> nc.TurningPoint:
    var_stats = ms.ExpandingMetricStats(pd.read_csv(filepath))
    return nc.TurningPoint.from_expanding_var_stats(var_stats)


def calculate_and_save_turning_points(
    config: types.RealConfig | types.PermutedConfig,
    read_directory: Path,
    save_directory: Path,
) -> None:
    tp_config = config["turning_point"]

    if not tp_config["should_calculate_it"]:
        return

    quantiles = utils.parse_value_or_iterable(tp_config["quantile"])
    metrics = utils.parse_value_or_iterable(tp_config["metric"])

    for quantile in quantiles:
        for metric in metrics:
            read_dir = read_directory / str(quantile) / metric
            filename_to_tp = utils.run_for_all_filenames(
                _calculate_turning_point,
                config["sports"],
                read_dir,
            )

            save_dir = save_directory / str(quantile) / metric
            utils.save_filename_to_df(filename_to_tp, save_dir)
