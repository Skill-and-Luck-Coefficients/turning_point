import logging

from config import parser, path
from logs import turning_logger

LOG_LEVEL = logging.INFO


def main() -> None:

    turning_logger.setLevel(LOG_LEVEL)

    configuration = parser.read_json_configuration("parameters.json")
    real_cfg = configuration["REAL_MATCHES"]
    perm_cfg = configuration["PERMUTED_MATCHES"]

    # SYNTHETIC MATCHES
    #   Permutation Matches
    parser.create_and_save_permuted_matches(
        config=perm_cfg,
        read_directory=path.MATCHES_PATH,
        save_directory=path.PERMUTED_MATCHES_PATH,
    )

    # COEFFICIENTS
    #   Variance Stats
    var_config_read_dir_save_dir = [
        (real_cfg, path.MATCHES_PATH, path.VARIANCE_STATS_PATH),
        (perm_cfg, path.PERMUTED_MATCHES_PATH, path.PERMUTED_VARIANCE_STATS_PATH),
    ]
    for variance_parameters in var_config_read_dir_save_dir:
        parser.calculate_and_save_var_stats(*variance_parameters)

    #   Turning Point
    tp_config_read_dir_save_dir = [
        (real_cfg, path.VARIANCE_STATS_PATH, path.TURNING_POINT_PATH),
        (perm_cfg, path.PERMUTED_VARIANCE_STATS_PATH, path.PERMUTED_TURNING_POINT_PATH),
    ]
    for turning_parameters in tp_config_read_dir_save_dir:
        parser.calculate_and_save_turning_points(*turning_parameters)


if __name__ == "__main__":
    main()