from typing import Any, Dict
from argparse import ArgumentParser, Namespace


class Arguments:
    """
    Handles command-line arguments parsing.
    """
    def __init__(self, argv: list[str]):
        self.argv = argv
        self.parser = ArgumentParser(
            description="DeepFake WebApp Control"
        )
        self._add_arguments()
        self.parsed_args = self._parse_args()

    def _add_arguments(self) -> None:
        self.parser.add_argument(
            "-v", "--version", 
            dest="version_main", 
            action="store_true", 
            help="Show version and exit"
        )

        # Create subparsers, but don't require them *yet*
        self.subparsers = self.parser.add_subparsers(dest="command")  # ← No required=True

        self.subparsers.add_parser("start", help="Start the web application")
        self.subparsers.add_parser("train", help="Start training the model")
        self.subparsers.add_parser("test", help="Start testing the model")
        self.subparsers.add_parser("extract", help="Start Extract Frames from Video Files to Images")

        self.subparsers.add_parser("create-userdir", help="Create user-data directory")

    def _parse_args(self) -> Namespace:
        args = self.parser.parse_args(self.argv)

        # Only enforce subcommand if version was not requested
        if not args.version_main and args.command is None:
            self.parser.error("a subcommand is required: {start, train, create-userdir}")

        return args

    def get_parsed_arg(self) -> Dict[str, Any]:
        return vars(self.parsed_args)