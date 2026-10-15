"""Run an ECS-versus-Mesa parity or profiling experiment.

Examples:
    uv run python -m experiments.benchmark_runner parity --replicates 100
    uv run python -m experiments.benchmark_runner profile --n-agents 2000 --steps 40
"""
import sys
import argparse
from collections.abc import Sequence

from experiments.parity.compare_implementations import main as run_parity
from experiments.profiling.profile_models import main as run_profile


def main(argv: Sequence[str] | None = None) -> None:
    arguments_list = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("experiment", choices=("parity", "profile"))
    if arguments_list == ["-h"] or arguments_list == ["--help"]:
        parser.print_help()
        return
    arguments, forwarded = parser.parse_known_args(arguments_list)

    if arguments.experiment == "parity":
        run_parity(forwarded)
    else:
        run_profile(forwarded)


if __name__ == "__main__":
    main()
