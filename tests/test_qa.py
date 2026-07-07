import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from common.generation import BASE_CITATION_FIELDS, METADATA_CITATION_FIELDS, build_source_snippets
from common.llm import build_messages
from common.qa import (
    RAGAnswerGenerator,
    extractive_answer,
    prediction_columns,
    query_filters_with_row_fallback,
    summarize_model_config,
)
from common.query import QueryFilters


def retrieved_chunk(text: str = "Debt was $100 million in March 2024.") -> dict:
    return {
        "score": 0.9,
        "text": text,
        "metadata": {
            "year": 2024,
            "month": 3,
            "source_path": "data/2024_03_treasurybulletin.txt",
            "heading": "Federal Debt",
            "content_type": "text",
            "chunk_id": "2024_03_00000",
        },
    }


class QATests(unittest.TestCase):
    def test_query_filters_prefer_source_metadata_over_question_dates(self):
        filters = query_filters_with_row_fallback(
            "How much was invested as of March 31, 2025?",
            [2024, 2025],
            row_year=2025,
            row_month=6,
        )

        self.assertEqual(filters, QueryFilters(year=2025, month=6))

    def test_query_filters_drop_question_month_when_source_month_is_ambiguous(self):
        filters = query_filters_with_row_fallback(
            "What was the change from June 2022 through September 2022?",
            [2022],
            row_year=2022,
            row_month=None,
        )

        self.assertEqual(filters, QueryFilters(year=2022, month=None))

    def test_extractive_answer_returns_evidence_sentence(self):
        answer = extractive_answer(
            "What happened to federal debt in March 2024?",
            [retrieved_chunk("Federal debt increased during March 2024. Unrelated sentence follows.")],
        )

        self.assertIn("Federal debt increased during March 2024.", answer)
        self.assertEqual(extractive_answer("What happened to federal debt?", []), "NOT_FOUND")

    def test_prediction_schema_and_model_config(self):
        self.assertEqual(
            prediction_columns(),
            [
                "question_id",
                "question",
                "gold_answer",
                "predicted_answer",
                "detected_year",
                "detected_month",
                "retrieval_method",
            ],
        )
        config = {
            "mode": "engineered",
            "metadata_enabled": True,
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k": 5,
            "embedding": {"model_name": "nvidia/llama-nemotron-embed-vl-1b-v2"},
        }

        summary = json.loads(summarize_model_config(config, "extractive", "extractive"))

        self.assertEqual(summary["chunk_size"], 512)
        self.assertTrue(summary["metadata_filtering"])
        self.assertEqual(summary["retrieval"], "faiss_vector_top_k")

    def test_source_snippets_can_use_base_or_metadata_citations(self):
        base = build_source_snippets([retrieved_chunk()], 500, BASE_CITATION_FIELDS)
        metadata = build_source_snippets([retrieved_chunk()], 500, METADATA_CITATION_FIELDS)
        messages = build_messages("What was debt in March 2024?", metadata)

        self.assertNotIn("Federal Debt", base[0].citation)
        self.assertIn("Federal Debt", metadata[0].citation)
        self.assertIn("Debt was $100 million", messages[1]["content"])
        self.assertIn("return exactly NOT_FOUND", messages[0]["content"])

    def test_deepinfra_generator_returns_model_answer(self):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Debt was $100 million. [S1]"))]
        )
        client = Mock()
        client.chat.completions.create.return_value = completion
        generator = RAGAnswerGenerator(
            {
                "metadata_enabled": True,
                "generation": {
                    "backend": "deepinfra",
                    "model": "test-model",
                    "base_url": "https://api.deepinfra.com/v1/openai",
                    "api_key_env": "DEEPINFRA_API_KEY",
                    "request_sleep_seconds": 0,
                    "retry_sleep_seconds": 0,
                },
            }
        )

        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "test-key"}), patch(
            "common.llm.OpenAI", return_value=client
        ) as openai_class:
            answer = generator.generate("What was debt in March 2024?", [retrieved_chunk()])

        self.assertEqual(answer, "Debt was $100 million. [S1]")
        self.assertEqual(openai_class.call_args.kwargs["api_key"], "test-key")
        self.assertEqual(openai_class.call_args.kwargs["base_url"], "https://api.deepinfra.com/v1/openai")
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["max_tokens"], 512)
        self.assertEqual(kwargs["timeout"], 90)
        self.assertIn("Retrieved context", kwargs["messages"][1]["content"])

    def test_generator_extractive_backend_stays_offline(self):
        generator = RAGAnswerGenerator({"metadata_enabled": False, "generation": {"backend": "extractive"}})

        answer = generator.generate(
            "What happened to federal debt in March 2024?",
            [retrieved_chunk("Federal debt increased during March 2024.")],
        )

        self.assertIn("Federal debt increased during March 2024.", answer)


if __name__ == "__main__":
    unittest.main()
