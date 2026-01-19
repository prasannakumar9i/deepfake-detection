"""system specific and performance tuning"""

from deepfake.system.gc_setup import gc_set_threshold
from deepfake.system.version_info import print_version_info


__all__ = ["asyncio_setup", "gc_set_threshold", "print_version_info"]
