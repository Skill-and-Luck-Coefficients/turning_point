from pathlib import Path

import pandas as pd

import turning_point.normal_coefficient as nc
import turning_point.variance_stats as vs
from logs import log, turning_logger

from .. import types


@log(turning_logger.info)
def _calculate_turning_point(
    filenames: str | list[str],
    read_directory: Path,
    tp_config: types.TurningPointConfig,
) -> dict[str, nc.TurningPoint]:
    if not tp_config["should_calculate_it"]:
        return {}

    filename_to_turning_point = {}

    for filename in filenames:
        filepath = read_directory / f"{filename}.csv"
        if not filepath.exists():
            turning_logger.warning(f"No file: {filepath}")
            continue

        var_stats = vs.ExpandingVarStats(pd.read_csv(filepath))

        # PermutationTurningPoints is the same when it comes to calculating it
        turning_point = nc.TurningPoint.from_expanding_var_stats(var_stats)
        filename_to_turning_point[filename] = turning_point

    return filename_to_turning_point


def calculate_and_save_turning_points(
    config: types.RealConfig | types.PermutedConfig,
    read_directory: Path,
    save_directory: Path,
) -> None:
    filename_to_turning_point = _calculate_turning_point(
        config["sports"],
        read_directory,
        config["turning_point"],
    )

    save_directory.mkdir(parents=True, exist_ok=True)
    for filename, turning_point in filename_to_turning_point.items():
        turning_point.df.to_csv(save_directory / f"{filename}.csv")
