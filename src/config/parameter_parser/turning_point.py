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
) -> dict[str, nc.TurningPoint]:
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


def _parse_quantiles(quantiles: float | list[float]) -> list[float]:
    if not isinstance(quantiles, list):
        quantiles = [quantiles]

    return quantiles


def _get_quantile_path(original_path: Path, quantile: float) -> Path:
    if quantile == 0.95:
        return original_path
    return original_path / str(quantile)


def calculate_and_save_turning_points(
    config: types.RealConfig | types.PermutedConfig,
    read_directory: Path,
    save_directory: Path,
) -> None:
    tp_config = config["turning_point"]

    if not tp_config["should_calculate_it"]:
        return

    quantiles = _parse_quantiles(tp_config["quantile"])

    for quantile in quantiles:
        quantile_read_dir = _get_quantile_path(read_directory, quantile)

        filename_to_turning_point = _calculate_turning_point(
            config["sports"],
            quantile_read_dir,
        )

        quantile_save_dir = _get_quantile_path(save_directory, quantile)
        quantile_save_dir.mkdir(parents=True, exist_ok=True)

        for filename, var_stats in filename_to_turning_point.items():
            var_stats.df.to_csv(quantile_save_dir / f"{filename}.csv")
