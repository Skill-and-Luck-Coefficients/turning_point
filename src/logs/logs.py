import logging
from pathlib import Path

NAME = "turning_point_logs"
FILE = Path("./turning_point_logs.log")
LEVEL = logging.WARNING

handler = logging.FileHandler(FILE)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

turning_logger = logging.getLogger(NAME)
turning_logger.setLevel(LEVEL)
turning_logger.addHandler(handler)
