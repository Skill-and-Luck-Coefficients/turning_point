from pathlib import Path

DATA_PATH = Path("../data/")

# Where to find tournament matches
MATCHES_PATH = DATA_PATH / "matches/real/"
PERMUTED_MATCHES_PATH = DATA_PATH / "matches/permuted/"
OPTIMAL_MATCHES_PATH = DATA_PATH / "matches/optimal/"
BT_MATCHES_PATH = DATA_PATH / "matches/bradley_terry/"

# Where to save coefficients and variances
TURNING_POINT_PATH = DATA_PATH / "turning_point/real/"
DIFF_POINTS_TURN_POINT_PATH = DATA_PATH / "turning_point/different_points/"
PERMUTED_TURNING_POINT_PATH = DATA_PATH / "turning_point/permuted/"
OPTIMAL_TURNING_POINT_PATH = DATA_PATH / "turning_point/optimal/"
BT_TURNING_POINT_PATH = DATA_PATH / "turning_point/bradley_terry/"

STATS_PATH = DATA_PATH / "stats/real/"
DIFF_POINTS_STATS_PATH = DATA_PATH / "stats/different_points/"
PERMUTED_STATS_PATH = DATA_PATH / "stats/permuted/"
OPTIMAL_STATS_PATH = DATA_PATH / "stats/optimal/"
BT_STATS_PATH = DATA_PATH / "stats/bradley_terry/"

# Where to save plots
PLOT_PATH = DATA_PATH / "images/"
