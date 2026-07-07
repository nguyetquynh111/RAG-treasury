import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from common.index import build_index
from common.vector_store import build_inner_product_index
from common.retrieval import VectorRetriever, matching_indices, ranked_vector_search


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


class RetrievalTests(unittest.TestCase):
    def test_ranked_vector_search_handles_filtered_indices(self):
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype="float32")
        index = build_inner_product_index(embeddings)

        ranked = ranked_vector_search(
            index=index,
            embeddings=embeddings,
            query=np.asarray([1.0, 0.0], dtype="float32"),
            allowed_indices=[1, 2],
            top_k=2,
        )

        self.assertEqual([index for index, _ in ranked], [2, 1])

    def test_matching_indices_accepts_multiple_date_pairs(self):
        chunks = [
            {"metadata": {"year": 2012, "month": 6}},
            {"metadata": {"year": 2022, "month": 6}},
            {"metadata": {"year": 2022, "month": 9}},
        ]

        matches = matching_indices(chunks, year=2022, month=6, date_pairs=((2012, 6), (2022, 6)))

        self.assertEqual(matches, [0, 1])

    def test_one_shared_index_serves_unfiltered_and_filtered_retrieval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_project(Path(tmpdir))
            with patch("common.index.Embedder", FakeEmbedder), patch("common.retrieval.Embedder", FakeEmbedder):
                result = build_index(config_path)
                baseline = VectorRetriever(config_path, mode="baseline")
                engineered = VectorRetriever(config_path, mode="engineered")
                baseline_results = baseline.retrieve("public debt", top_k=2)
                engineered_results = engineered.retrieve("public debt", top_k=2, date_pairs=((2012, 6),))

        self.assertIn("outputs/index/index.faiss", result["index_path"])
        self.assertEqual(len(baseline_results), 2)
        self.assertEqual(len(engineered_results), 1)
        self.assertEqual(engineered_results[0]["metadata"]["year"], 2012)
        self.assertEqual(engineered_results[0]["metadata"]["heading"], "Older Bulletin")

    def _write_project(self, root: Path) -> Path:
        data_dir = root / "data"
        config_dir = root / "config"
        data_dir.mkdir()
        config_dir.mkdir()
        (data_dir / "2024_03_treasurybulletin.txt").write_text(
            "# March Bulletin\npublic debt treasury securities", encoding="utf-8"
        )
        (data_dir / "2012_06_treasurybulletin.txt").write_text(
            "# Older Bulletin\nolder public debt reference", encoding="utf-8"
        )
        config_path = config_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "data_dir": "data",
                    "csv_path": "officeqa_full.csv",
                    "selected_years": [2024],
                    "document_years": [2012, 2024],
                    "vector_db_type": "faiss",
                    "index_dir": "outputs/index",
                    "embedding": {"backend": "deepinfra"},
                    "chunk_size": 8,
                    "chunk_overlap": 0,
                    "top_k": 2,
                    "runs": {
                        "baseline": {"output_dir": "outputs/baseline", "metadata_enabled": False},
                        "engineered": {"output_dir": "outputs/engineered", "metadata_enabled": True},
                    },
                }
            ),
            encoding="utf-8",
        )
        return config_path


if __name__ == "__main__":
    unittest.main()
