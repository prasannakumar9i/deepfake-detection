import logging
import shutil
from pathlib import Path

from deepfake.constants import Config, USER_DATA_FILES
from deepfake.exceptions import OperationalException


logger = logging.getLogger(__name__)


def create_datadir(config: dict, datadir: str | None = None) -> Path:
    # Determine the base data folder
    folder = Path(datadir) if datadir else Path(f"{config['user_data_dir']}/data")
    
    # Create the main data directory if it doesn't exist
    if not folder.is_dir():
        folder.mkdir(parents=True)
        logger.info(f"Created data directory: {folder}")

    # Create 'real' and 'fake' subdirectories
    real_dir = folder / "original"
    fake_dir = folder / "manipulated"

    for subfolder in [real_dir, fake_dir]:
        if not subfolder.is_dir():
            subfolder.mkdir(parents=True)
            logger.info(f"Created subdirectory: {subfolder}")

    return folder


def create_img_dir(config: dict, img_dir: str | None = None) -> Path:
    # Determine the base data folder
    folder = Path(img_dir) if img_dir else Path(f"{config['user_data_dir']}/images")
    
    # Create the main data directory if it doesn't exist
    if not folder.is_dir():
        folder.mkdir(parents=True)
        logger.info(f"Created data directory: {folder}")

    # Create subdirectories
    test_dir = folder / "test"
    train_dir = folder / "train"
    temp_dir = folder / "temp"

    for subfolder in [test_dir, train_dir, temp_dir]:
        if not subfolder.is_dir():
            subfolder.mkdir(parents=True)
            logger.info(f"Created subdirectory: {subfolder}")

    return folder

def create_dir(config: dict, dir_name: str | None = None) -> Path:
    # Determine the base data folder
    folder = Path(dir_name)
    
    # Create the main data directory if it doesn't exist
    if not folder.is_dir():
        folder.mkdir(parents=True)
        logger.info(f"Created directory: {folder}")

    return folder



def create_userdata_dir(directory: str, create_dir: bool = False) -> Path:
    sub_dirs = [
        "results",
        "data",
        "models",
        "uploads",
        "logs",
    ]
    folder = Path(directory)

    if not folder.is_dir():
        if create_dir:
            folder.mkdir(parents=True)
            logger.info(f"Created user-data directory: {folder}")
        else:
            raise OperationalException(
                f"Directory `{folder}` does not exist. "
                "Please use `deepfake create-userdir` to create a user directory"
            )

    # Create required subdirectories
    for f in sub_dirs:
        subfolder = folder / f
        if not subfolder.is_dir():
            if subfolder.exists() or subfolder.is_symlink():
                raise OperationalException(
                    f"File `{subfolder}` exists already and is not a directory. "
                    "Freqtrade requires this to be a directory."
                )
            subfolder.mkdir(parents=False)
    return folder

def copy_sample_files(directory: Path, overwrite: bool = False) -> None:
    """
    Copy files from templates to User data directory.
    :param directory: Directory to copy data to
    :param overwrite: Overwrite existing sample files
    """
    if not directory.is_dir():
        raise OperationalException(f"Directory `{directory}` does not exist.")
    sourcedir = Path(__file__).parents[1] / "templates"
    shutil.copy(str(sourcedir / "config.json"), str(directory / "config.json"))
    
    for source, target in USER_DATA_FILES.items():
        targetdir = directory / target
        if not targetdir.is_dir():
            raise OperationalException(f"Directory `{targetdir}` does not exist.")
        targetfile = targetdir / source
        if targetfile.exists():
            if not overwrite:
                logger.warning(f"File `{targetfile}` exists already, not deploying sample file.")
                continue
            logger.warning(f"File `{targetfile}` exists already, overwriting.")
        shutil.copy(str(sourcedir / source), str(targetfile))

