"""Command-line adapter for the Deep ALM reproduction runner."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from deepalm.config import ConfigurationError, load_configuration
from deepalm.runner import ReproductionRunner, RunStatus


def main(argv: Sequence[str] | None = None) -> int:
    """Run a configured reproduction and return a process exit status."""

    parser = argparse.ArgumentParser(prog="deepalm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="create an auditable run bundle")
    run_parser.add_argument(
        "--config", type=Path, required=True, help="YAML run configuration"
    )
    arguments = parser.parse_args(argv)

    try:
        configuration = load_configuration(arguments.config)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    bundle = ReproductionRunner().run(configuration)
    if bundle.status is RunStatus.COMPLETED:
        assert bundle.artifact_directory is not None
        print(bundle.artifact_directory)
        return 0
    if bundle.status is RunStatus.ACCEPTANCE_FAILED:
        print(
            "Run completed but did not meet acceptance requirements.", file=sys.stderr
        )
        return 3

    print(f"Operational failure: {bundle.error}", file=sys.stderr)
    return 1
