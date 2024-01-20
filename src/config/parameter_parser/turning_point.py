from pathlib import Path
from typing import TypeVar

import pandas as pd

import turning_point.metric_stats as ms
import turning_point.normal_coefficient as nc
from logs import log, turning_logger

from .. import types

T = TypeVar("T")


@log(turning_logger.info)
def _calculate_turning_point(
    filenames: str | list[str],
    read_directory: Path,
) -> dict[str, nc.TurningPoint]:
    filename_to_turning_point = {}

    for filename in filenames:
        filepath = read_directory / f"{filename}.csv"
        if not filepath.exists():
            turning_logger.warning(f"No file: {filepath}")
            continue

        var_stats = ms.ExpandingMetricStats(pd.read_csv(filepath))

        # PermutationTurningPoints is the same when it comes to calculating it
        turning_point = nc.TurningPoint.from_expanding_var_stats(var_stats)
        filename_to_turning_point[filename] = turning_point

    return filename_to_turning_point


def _parse_parameter(parameter: T | list[T]) -> list[T]:
    if not isinstance(parameter, list):
        parameter = [parameter]
    return parameter


def calculate_and_save_turning_points(
    config: types.RealConfig | types.PermutedConfig,
    read_directory: Path,
    save_directory: Path,
) -> None:
    tp_config = config["turning_point"]

    if not tp_config["should_calculate_it"]:
        return

    quantiles = _parse_parameter(tp_config["quantile"])
    metrics = _parse_parameter(tp_config["metric"])

    for quantile in quantiles:
        for metric in metrics:
            read_dir = read_directory / str(quantile) / metric
            filename_to_tp = _calculate_turning_point(config["sports"], read_dir)

            save_dir = save_directory / str(quantile) / metric
            save_dir.mkdir(parents=True, exist_ok=True)

            for filename, var_stats in filename_to_tp.items():
                var_stats.df.to_csv(save_dir / f"{filename}.csv")
