"""Inspect the Treasury document dataset selected by config."""

from __future__ import annotations

import argparse
from collections import Counter

from common.config import DEFAULT_CONFIG_PATH, load_index_config, resolve_path
from common.dataset import load_treasury_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect selected Treasury .txt files.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to shared YAML config.")
    args = parser.parse_args()

    config = load_index_config(args.config)
    data_dir = resolve_path(args.config, config["data_dir"])
    documents = load_treasury_documents(data_dir, config["document_years"])

    by_year = Counter(document.year for document in documents)
    by_month = Counter((document.year, document.month) for document in documents)
    print(f"Loaded {len(documents)} Treasury .txt documents from {data_dir}")
    print(f"Years: {dict(sorted(by_year.items()))}")
    print(f"Year/months: {dict(sorted(by_month.items()))}")


if __name__ == "__main__":
    main()
