import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from engineered.generation import RAGAnswerGenerator, build_messages, build_source_snippets
from engineered.qa import grounded_answer, prediction_columns, summarize_model_config
from engineered.retrieval import Candidate


class EngineeredQATests(unittest.TestCase):
    def test_grounded_answer_returns_evidence_sentence(self):
        candidate = Candidate(
            text="Federal debt increased during March 2024. Unrelated sentence follows.",
            metadata={
                "year": 2024,
                "month": 3,
                "source_path": "data/2024_03_treasurybulletin.txt",
                "heading": "Federal Debt",
                "chunk_id": "2024_03_0000_0000",
            },
            vector_score=0.9,
            bm25_score=2.0,
            rerank_score=0.8,
        )

        answer = grounded_answer("What happened to federal debt in March 2024?", [candidate])

        self.assertIn("Federal debt increased during March 2024.", answer)

    def test_grounded_answer_returns_not_found_without_evidence(self):
        candidate = Candidate(
            text="International exchange-rate tables are listed here.",
            metadata={
                "year": 2024,
                "month": 3,
                "source_path": "data/2024_03_treasurybulletin.txt",
                "heading": "Exchange Rates",
                "chunk_id": "2024_03_0000_0000",
            },
        )

        self.assertEqual(grounded_answer("What happened to federal debt?", [candidate]), "NOT_FOUND")
        self.assertEqual(grounded_answer("What happened to federal debt?", []), "NOT_FOUND")

    def test_prediction_schema_and_model_config(self):
        self.assertEqual(
            prediction_columns(),
            [
                "question_id",
                "question",
                "gold_answer",
                "predicted_answer",
                "selected_years",
                "detected_year",
                "detected_month",
                "retrieved_sources",
                "retrieved_context_ids",
                "retrieved_context",
                "retrieval_method",
                "model_config",
            ],
        )
        config = {
            "chunk_size": 768,
            "chunk_overlap": 100,
            "bm25_top_k": 20,
            "vector_top_k": 20,
            "final_top_k": 5,
            "embedding": {"model_name": "nvidia/llama-nemotron-embed-vl-1b-v2"},
        }

        summary = json.loads(summarize_model_config(config, "fallback"))

        self.assertEqual(summary["chunk_size"], 768)
        self.assertEqual(summary["reranking_method"], "fallback")

    def test_generator_builds_grounded_prompt_with_sources(self):
        candidate = Candidate(
            text="Debt was $100 million in March 2024.",
            metadata={
                "year": 2024,
                "month": 3,
                "source_path": "data/2024_03_treasurybulletin.txt",
                "heading": "Federal Debt",
                "chunk_id": "2024_03_0000_0000",
            },
        )

        snippets = build_source_snippets([candidate], max_context_chars=500)
        messages = build_messages("What was debt in March 2024?", snippets)

        self.assertIn("[S1]", messages[1]["content"])
        self.assertIn("Debt was $100 million", messages[1]["content"])
        self.assertIn("return exactly NOT_FOUND", messages[0]["content"])

    def test_deepinfra_generator_returns_model_answer(self):
        candidate = Candidate(
            text="Debt was $100 million in March 2024.",
            metadata={
                "year": 2024,
                "month": 3,
                "source_path": "data/2024_03_treasurybulletin.txt",
                "heading": "Federal Debt",
                "chunk_id": "2024_03_0000_0000",
            },
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Debt was $100 million. [S1]"))]
        )
        client = Mock()
        client.chat.completions.create.return_value = completion
        generator = RAGAnswerGenerator(
            {
                "generation": {
                    "backend": "deepinfra",
                    "model": "test-model",
                    "base_url": "https://api.deepinfra.com/v1/openai",
                    "api_key_env": "DEEPINFRA_API_KEY",
                    "request_sleep_seconds": 0,
                    "retry_sleep_seconds": 0,
                }
            },
            extractive_fallback=grounded_answer,
        )

        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "test-key"}), patch(
            "common.rag_generation.OpenAI", return_value=client
        ) as openai_class:
            answer = generator.generate("What was debt in March 2024?", [candidate])

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
        candidate = Candidate(
            text="Federal debt increased during March 2024.",
            metadata={
                "year": 2024,
                "month": 3,
                "source_path": "data/2024_03_treasurybulletin.txt",
                "heading": "Federal Debt",
                "chunk_id": "2024_03_0000_0000",
            },
            rerank_score=0.8,
        )
        generator = RAGAnswerGenerator(
            {"generation": {"backend": "extractive"}},
            extractive_fallback=grounded_answer,
        )

        answer = generator.generate("What happened to federal debt in March 2024?", [candidate])

        self.assertIn("Federal debt increased during March 2024.", answer)


if __name__ == "__main__":
    unittest.main()
