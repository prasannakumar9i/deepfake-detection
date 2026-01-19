from typing import Any

DEFAULT_CONFIG = "config.json"
DEFAULT_DB_PROD_URL = "sqlite:///user_data/datav3.sqlite"
DEFAULT_LOG_FILE = "deepfake.log"

DATETIME_PRINT_FORMAT = "%Y-%m-%d %H:%M:%S"

USER_DATA_DIR = "user_data"
DATA_DIR = "data_dir"
UPLOAD_DIR = "upload_dir"

# Source files with destination directories within user-directory
USER_DATA_FILES = {
    "shape_predictor_81_face_landmarks.dat": "models"
}

FACE_PREDICTOR_NAME = "shape_predictor_81_face_landmarks.dat"

Config = dict[str, Any]
