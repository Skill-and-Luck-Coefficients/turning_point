import json
from pathlib import Path

from .. import types


def read_json_configuration(path: Path) -> types.ConfigurationType:
    with open(path, "r") as config_file:
        configuration = json.load(config_file)

    return configuration
