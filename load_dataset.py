"""Inspect the baseline Treasury document dataset."""

from __future__ import annotations

import argparse
from collections import Counter

from baseline.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from baseline.dataset import load_treasury_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Treasury .txt files for the baseline years.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to baseline YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = resolve_path(args.config, config["data_dir"])
    documents = load_treasury_documents(data_dir, config["selected_years"])

    by_year = Counter(document.year for document in documents)
    by_month = Counter((document.year, document.month) for document in documents)
    print(f"Loaded {len(documents)} Treasury .txt documents from {data_dir}")
    print(f"Years: {dict(sorted(by_year.items()))}")
    print(f"Year/months: {dict(sorted(by_month.items()))}")


if __name__ == "__main__":
    main()
