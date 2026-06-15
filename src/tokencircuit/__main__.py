import argparse
import sys

from . import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tokencircuit",
        description="Detect infinite loops in LLM agentic workflows",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"tokencircuit {__version__}",
    )
    args = parser.parse_args()
    _ = args


if __name__ == "__main__":
    main()
