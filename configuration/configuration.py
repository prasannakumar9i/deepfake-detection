"""
This module contains the configuration class
"""

import logging
from pathlib import Path
from typing import Any

from deepfake import constants
from deepfake.configuration.directory_operations import (
    create_datadir, 
    create_userdata_dir,
    create_dir,
    create_img_dir
)
from deepfake.configuration.load_config import load_file
from deepfake.constants import Config

from deepfake.loggers import setup_logging
from deepfake.misc import parse_db_uri_for_logging


logger = logging.getLogger(__name__)


class Configuration:
    """
    Class to read and init the bot configuration
    Reuse this class for the bot, backtesting, hyperopt and every script that required configuration
    """

    def __init__(self) -> None:
        self.config: Config | None = None

    def get_config(self) -> Config:
        """
        Return the config. Use this method to get the bot config
        :return: Dict: Bot config
        """
        if self.config is None:
            self.config = self.load_config()

        return self.config

    def load_config(self) -> dict[str, Any]:
        """
        Extract information for sys.argv and load the bot configuration
        :return: Configuration dictionary
        """
        # Load all configs
        config: Config = {}

        file_path = Path("user_data") / "config.json"
        config: Config = load_file(file_path)

        self._process_logging(config)

        self._process_db(config)
        
        self._process_datadir(config)
        
        return config

    def _process_logging(self, config: Config) -> None:
         # Log level
        if "verbosity" not in config:
            config.update({"verbosity": 0})

        # Log level
        config.update({"logfile": f"user_data/logs/{constants.DEFAULT_LOG_FILE}"})
        config.update({"print_colorized": True})

        setup_logging(config)

    def _process_db(self, config: Config) -> None:

        config["db_url"] = constants.DEFAULT_DB_PROD_URL
        logger.info(f'Using DB: "{parse_db_uri_for_logging(config["db_url"])}"')


    def _process_datadir(self, config: Config) -> None:

        config.update({"user_data_dir": str(Path.cwd() / "user_data")})

        # reset to user_data_dir so this contains the absolute path.
        config["user_data_dir"] = create_userdata_dir(config["user_data_dir"], create_dir=False)
        logger.info("Using user-data directory: %s ...", config["user_data_dir"])

        config.update({"datadir": create_datadir(config)})
        logger.info("Using data directory: %s ...", config.get("datadir"))

        config.update({"imgdir": create_img_dir(config)})
        logger.info("Using img directory: %s ...", config.get("imgdir"))
        
        config.update({"modelsdir": config["user_data_dir"] / "models"})
        logger.info("Using models directory: %s ...", config.get("modelsdir"))
        
        # uplaod path 
        config.update({"upload_dir": config["user_data_dir"] / "uploads"})