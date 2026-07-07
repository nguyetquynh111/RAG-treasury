from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from common.config import load_pipeline_config, resolve_path


class ConfigTests(unittest.TestCase):
    def _write_config(self, root: Path, overrides: dict | None = None) -> Path:
        config = {
            "data_dir": "data",
            "csv_path": "officeqa_full.csv",
            "selected_years": [2022, "2023"],
            "vector_db_type": "faiss",
            "chunk_size": 12,
            "chunk_overlap": 2,
            "top_k": 3,
            "runs": {
                "baseline": {"output_dir": "outputs/baseline", "metadata_enabled": False},
                "engineered": {"output_dir": "outputs/engineered", "metadata_enabled": True},
            },
        }
        if overrides:
            config.update(overrides)

        config_dir = root / "config"
        config_dir.mkdir()
        config_path = config_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return config_path

    def test_load_pipeline_config_selects_only_run_output_and_metadata_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(root)
            baseline = load_pipeline_config(config_path, mode="baseline")
            engineered = load_pipeline_config(config_path, mode="engineered")
            resolved_data = resolve_path(config_path, baseline["data_dir"])

        self.assertEqual(baseline["selected_years"], [2022, 2023])
        self.assertEqual(baseline["document_years"], [2022, 2023])
        self.assertEqual(baseline["output_dir"], "outputs/baseline")
        self.assertFalse(baseline["metadata_enabled"])
        self.assertEqual(engineered["output_dir"], "outputs/engineered")
        self.assertTrue(engineered["metadata_enabled"])
        self.assertEqual(resolved_data, (root / "data").resolve())

    def test_rejects_per_run_algorithm_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(
                Path(tmpdir),
                {
                    "runs": {
                        "baseline": {"output_dir": "outputs/baseline", "metadata_enabled": False, "top_k": 1},
                        "engineered": {"output_dir": "outputs/engineered", "metadata_enabled": True},
                    }
                },
            )

            with self.assertRaisesRegex(ValueError, "Unsupported per-run keys"):
                load_pipeline_config(config_path, mode="baseline")


if __name__ == "__main__":
    unittest.main()
