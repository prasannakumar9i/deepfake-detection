#!/usr/bin/env python3
"""
Main Deepfake bot script.
Run with --help to see available commands.
"""

import logging
import sys

from deepfake.arguments import Arguments
from deepfake.deepfake import (
    DeepFake,
    start_create_userdir
)
from deepfake.exceptions import DeepfakeException
from deepfake.loggers import setup_logging_pre
from deepfake.system import gc_set_threshold, print_version_info

logger = logging.getLogger("deepfake")


def main() -> None:
    """
    Initiate the DeepFake bot and start the appropriate command.
    """
    return_code: int = 1

    try:
        setup_logging_pre()
        gc_set_threshold()

        arguments = Arguments(sys.argv[1:])
        args = arguments.get_parsed_arg()

        # Handle global version flag
        if args.get("version_main"):
            print_version_info()
            return_code = 0

        # Command dispatch
        elif args.get("command") == "start":
            DeepFake().startup()

        elif args.get("command") == "train":
            DeepFake().start_train()

        elif args.get("command") == "test":
            DeepFake().start_test()
            
        elif args.get("command") == "extract":
            DeepFake().start_extract_frames()
            
        elif args.get("command") == "create-userdir":
            start_create_userdir()

    except SystemExit as e:
        return_code = e.code
    except KeyboardInterrupt:
        logger.info("SIGINT received, aborting...")
        return_code = 0
    except DeepfakeException as e:
        logger.error(str(e))
        return_code = 2
    except Exception:
        logger.exception("Fatal exception!")
    finally:
        sys.exit(return_code)


if __name__ == "__main__":  # pragma: no cover
    main()
