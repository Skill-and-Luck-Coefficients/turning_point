from pathlib import Path

DATA_PATH = Path("../data/")

# Where to find tournament matches
MATCHES_PATH = DATA_PATH / "matches/real/"
PERMUTED_MATCHES_PATH = DATA_PATH / "matches/permuted/"
OPTIMAL_MATCHES_PATH = DATA_PATH / "matches/optimal/"

# Where to save coefficients and variances
TURNING_POINT_PATH = DATA_PATH / "turning_point/real/"
PERMUTED_TURNING_POINT_PATH = DATA_PATH / "turning_point/permuted/"
OPTIMAL_TURNING_POINT_PATH = DATA_PATH / "turning_point/optimal/"

VARIANCE_STATS_PATH = DATA_PATH / "variance_stats/real/"
PERMUTED_VARIANCE_STATS_PATH = DATA_PATH / "variance_stats/permuted/"
OPTIMAL_VARIANCE_STATS_PATH = DATA_PATH / "variance_stats/optimal/"

# Where to save plots
PLOT_PATH = DATA_PATH / "images/"
