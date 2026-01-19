from deepfake import __version__


def print_version_info():
    """Print version information for deepfake and its key dependencies."""
    import platform
    import sys

    print(f"Operating System:\t{platform.platform()}")
    print(f"Python Version:\t\tPython {sys.version.split(' ')[0]}")
    print()
    print(f"Deepfake Version:\tdeepfake {__version__}")
