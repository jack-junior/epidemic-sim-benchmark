"""Compatibility entry point for the experiment benchmark runner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.benchmark_runner import main


if __name__ == "__main__":
    main()
