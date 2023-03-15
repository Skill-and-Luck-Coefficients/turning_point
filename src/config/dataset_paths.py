from pathlib import Path

DATA_PATH = Path("../data/")

# Where to find tournament matches
MATCHES_PATH = DATA_PATH / "matches/real/"
PERMUTED_MATCHES_PATH = DATA_PATH / "matches/permuted/"

# Where to save coefficients and variances
TURNING_POINT_PATH = DATA_PATH / "turning_point/real/"
PERMUTED_TURNING_POINT_PATH = DATA_PATH / "turning_point/permuted/"

VARIANCE_STATS_PATH = DATA_PATH / "variance_stats/real/"
PERMUTED_VARIANCE_STATS_PATH = DATA_PATH / "variance_stats/permuted/"

# Where to save plots
PLOT_PATH = DATA_PATH / "images/"
