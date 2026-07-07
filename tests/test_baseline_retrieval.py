import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from baseline.index import build_index
from baseline.retrieval import Retriever, load_chunks


class FakeEmbedder:
    actual_backend = "deepinfra"
    model_name = "nvidia/llama-nemotron-embed-vl-1b-v2"

    def __init__(self, config):
        self.config = config

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("public" in lowered or "debt" in lowered or "securities" in lowered),
                    float("cash" in lowered or "deposits" in lowered),
                    float("older" in lowered),
                ]
            )
        matrix = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


class BaselineRetrievalTests(unittest.TestCase):
    def _write_temp_project(self, root: Path) -> Path:
        data_dir = root / "data"
        config_dir = root / "config"
        data_dir.mkdir()
        config_dir.mkdir()

        (data_dir / "2024_03_treasurybulletin.txt").write_text(
            "public debt treasury securities March borrowing fiscal receipts",
            encoding="utf-8",
        )
        (data_dir / "2024_06_treasurybulletin.txt").write_text(
            "cash balance June account deposits operating cash",
            encoding="utf-8",
        )
        (data_dir / "2023_03_treasurybulletin.txt").write_text(
            "older March record not selected",
            encoding="utf-8",
        )

        config = {
            "data_dir": "data",
            "csv_path": "officeqa_full.csv",
            "selected_years": [2024],
            "vector_db_type": "faiss",
            "embedding": {
                "backend": "deepinfra",
                "model_name": "nvidia/llama-nemotron-embed-vl-1b-v2",
                "normalize": True,
            },
            "chunk_size": 8,
            "chunk_overlap": 0,
            "top_k": 2,
            "output_dir": "outputs/baseline",
        }
        config_path = config_dir / "baseline.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return config_path

    def test_build_index_and_retrieve_with_year_month_filters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_temp_project(Path(tmpdir))

            with patch("baseline.index.Embedder", FakeEmbedder), patch(
                "baseline.retrieval.Embedder", FakeEmbedder
            ):
                result = build_index(config_path)
                retriever = Retriever(config_path)
                march_results = retriever.retrieve("public debt securities", year=2024, month=3, top_k=1)
                no_results = retriever.retrieve("public debt securities", year=2023, month=3, top_k=1)

            chunks = load_chunks(result["chunks_path"])

        self.assertEqual(result["documents"], 2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(march_results), 1)
        self.assertEqual(march_results[0]["metadata"]["year"], 2024)
        self.assertEqual(march_results[0]["metadata"]["month"], 3)
        self.assertEqual(no_results, [])


if __name__ == "__main__":
    unittest.main()
