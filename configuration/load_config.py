"""
This module contain functions to load the configuration file
"""

import logging
from pathlib import Path
from typing import Any

import rapidjson

from deepfake.exceptions import OperationalException


logger = logging.getLogger(__name__)


CONFIG_PARSE_MODE = rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS


def load_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r") as file:
            config = rapidjson.load(file, parse_mode=CONFIG_PARSE_MODE)
    except FileNotFoundError:
        raise OperationalException(f'File "{path}" not found!') from None
    return config
