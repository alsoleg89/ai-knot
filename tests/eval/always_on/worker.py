"""Child process for declared crash checkpoints (intentionally bypasses cleanup)."""

import sys
from pathlib import Path

from .scenarios import crash_worker

if __name__ == "__main__":
    crash_worker(Path(sys.argv[1]))
