"""Build and save the engineered FAISS retrieval index."""

from __future__ import annotations

import argparse
import json

from engineered.config import DEFAULT_CONFIG_PATH
from engineered.retrieval import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the engineered Treasury retrieval index.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to engineered YAML config.")
    args = parser.parse_args()

    result = build_index(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
