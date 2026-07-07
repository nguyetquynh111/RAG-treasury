from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from baseline.config import load_config, resolve_path


class BaselineConfigTests(unittest.TestCase):
    def _write_config(self, root: Path, overrides: dict | None = None) -> Path:
        config = {
            "data_dir": "data",
            "csv_path": "officeqa_full.csv",
            "selected_years": [2022, "2023"],
            "vector_db_type": "faiss",
            "chunk_size": 12,
            "chunk_overlap": 2,
            "top_k": 3,
            "output_dir": "outputs/baseline",
        }
        if overrides:
            config.update(overrides)

        config_dir = root / "config"
        config_dir.mkdir()
        config_path = config_dir / "baseline.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return config_path

    def test_load_config_coerces_values_and_resolves_project_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(root)

            config = load_config(config_path)
            resolved_data = resolve_path(config_path, config["data_dir"])

        self.assertEqual(config["selected_years"], [2022, 2023])
        self.assertEqual(config["chunk_size"], 12)
        self.assertEqual(config["chunk_overlap"], 2)
        self.assertEqual(config["top_k"], 3)
        self.assertEqual(resolved_data, (root / "data").resolve())

    def test_load_config_rejects_invalid_chunk_overlap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(
                Path(tmpdir),
                {"chunk_size": 10, "chunk_overlap": 10},
            )

            with self.assertRaisesRegex(ValueError, "chunk_overlap"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
