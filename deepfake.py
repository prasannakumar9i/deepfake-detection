"""
The Main class
"""

import logging

from deepfake.constants import Config
from deepfake.configuration import Configuration
from deepfake.persistence import init_db

from deepfake.deepfakeai import Predictor

logger = logging.getLogger(__name__)


class DeepFake:
    """
    Main Class
    """
    def __init__(self):
        self.config: Config = Configuration().get_config()
        init_db(self.config["db_url"])
        
    def startup(self) -> None:
        """
        Start the Webserver to test.
        """
        from deepfake.rpc.api_server.webserver import ApiServer
        from deepfake.rpc import RPC
        
        self.predictor = Predictor(self.config)

        # Start the Websrver
        self._api_server = ApiServer(self.config)
        self._rpc = RPC(self)
        
        self._api_server.add_rpc_handler(self._rpc)
        
    def start_train(self) -> None:
        """"
        strat the training model...
        """
        from deepfake.deepfakeai import train_model
        train_model(self.config)
        
    def start_test(self) -> None:
        """"
        strat the training model...
        """
        from deepfake.deepfakeai import test_model
        test_model(self.config)
    
    def start_extract_frames(self,) -> None:
        """"
        strat the training model...
        """
        from deepfake.deepfakeai import extract_frames

        extract_frames(self.config)
    

def start_create_userdir() -> None:
    """
    Create "user_data" directory to contain user data.
    """
    from deepfake.configuration.directory_operations import (
        create_userdata_dir,
        copy_sample_files
    )
    from deepfake.constants import USER_DATA_DIR
    
    user_dir = create_userdata_dir(USER_DATA_DIR, create_dir=True)
    copy_sample_files(user_dir)