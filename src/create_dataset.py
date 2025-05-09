import logging
from pathlib import Path

from config import parser, path
from logs import turning_logger

LOG_LEVEL = logging.INFO


def main() -> None:
    turning_logger.setLevel(LOG_LEVEL)

    json_path = Path("parameters.json")
    configuration = parser.read_parameters.read_json_configuration(json_path)

    real_cfg = configuration["REAL_MATCHES"]
    perm_cfg = configuration["PERMUTED_MATCHES"]
    opt_cfg = configuration["OPTIMAL_SCHEDULE"]
    diff_cfg = configuration["DIFFERENT_POINT_SYSTEM"]
    bt_cfg = configuration["BRADLEY_TERRY"]

    # SYNTHETIC MATCHES
    #   Permutation Matches
    parser.permutations.create_and_save_permuted_matches(
        config=perm_cfg,
        read_directory=path.MATCHES_PATH,
        save_directory=path.PERMUTED_MATCHES_PATH,
    )

    #   Optimal Schedule
    parser.optimal_schedule.create_and_save_optimal_matches(
        config=opt_cfg,
        read_directory=path.MATCHES_PATH,
        save_directory=path.OPTIMAL_MATCHES_PATH,
    )

    #   Optimal Schedule Validation
    parser.bradley_terry.create_and_save_bradltey_terry_matches(
        config=bt_cfg,
        save_directory=path.BT_MATCHES_PATH,
    )

    # COEFFICIENTS
    #   Variance Stats
    var_config_read_dir_save_dir = [
        (real_cfg, path.MATCHES_PATH, path.STATS_PATH),
        (diff_cfg, path.MATCHES_PATH, path.DIFF_POINTS_STATS_PATH),
        (perm_cfg, path.PERMUTED_MATCHES_PATH, path.PERMUTED_STATS_PATH),
        (opt_cfg, path.OPTIMAL_MATCHES_PATH, path.OPTIMAL_STATS_PATH),
        (bt_cfg, path.BT_MATCHES_PATH, path.BT_STATS_PATH),
    ]
    for variance_parameters in var_config_read_dir_save_dir:
        parser.metrics.calculate_and_save_metric_stats(*variance_parameters)

    #   Turning Point
    tp_config_read_dir_save_dir = [
        (real_cfg, path.STATS_PATH, path.TURNING_POINT_PATH),
        (diff_cfg, path.DIFF_POINTS_STATS_PATH, path.DIFF_POINTS_TURN_POINT_PATH),
        (perm_cfg, path.PERMUTED_STATS_PATH, path.PERMUTED_TURNING_POINT_PATH),
        (opt_cfg, path.OPTIMAL_STATS_PATH, path.OPTIMAL_TURNING_POINT_PATH),
        (bt_cfg, path.BT_STATS_PATH, path.BT_TURNING_POINT_PATH),
    ]
    for turning_parameters in tp_config_read_dir_save_dir:
        parser.turning_point.calculate_and_save_turning_points(*turning_parameters)


if __name__ == "__main__":
    main()
