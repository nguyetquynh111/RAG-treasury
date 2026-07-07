import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from common.embeddings import Embedder


class DeepInfraEmbeddingTests(unittest.TestCase):
    def test_embedder_calls_deepinfra_embeddings_and_normalizes_vectors(self):
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 3.0, 4.0]),
                SimpleNamespace(index=0, embedding=[3.0, 4.0, 0.0]),
            ]
        )
        config = {
            "embedding": {
                "backend": "deepinfra",
                "model_name": "nvidia/llama-nemotron-embed-vl-1b-v2",
                "base_url": "https://api.deepinfra.com/v1/openai",
                "api_key_env": "DEEPINFRA_API_KEY",
                "batch_size": 2,
                "timeout_seconds": 30,
                "normalize": True,
            }
        }

        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "test-key"}), patch(
            "common.embeddings.OpenAI", return_value=client
        ) as openai_class:
            vectors = Embedder(config).encode(["alpha", "beta"])

        openai_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.deepinfra.com/v1/openai",
        )
        client.embeddings.create.assert_called_once_with(
            model="nvidia/llama-nemotron-embed-vl-1b-v2",
            input=["alpha", "beta"],
            timeout=30,
        )
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(vectors[0], np.array([0.6, 0.8, 0.0], dtype="float32"))
        np.testing.assert_allclose(vectors[1], np.array([0.0, 0.6, 0.8], dtype="float32"))

    def test_rejects_non_deepinfra_backend(self):
        with self.assertRaisesRegex(ValueError, "embedding.backend"):
            Embedder({"embedding": {"backend": "local"}})


if __name__ == "__main__":
    unittest.main()
