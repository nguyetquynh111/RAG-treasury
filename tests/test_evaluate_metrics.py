import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from evaluate import metrics
from evaluate.metrics import (
    ChunkRecord,
    answer_matches_gold,
    correct_documents_found,
    evaluate_predictions,
    extract_year_month_pairs,
    first_relevant_rank,
    judge_bool,
    judge_count,
    relevant_chunk_ids,
    safe_divide,
)


class EvaluateMetricsTests(unittest.TestCase):
    def test_extract_year_month_pairs_from_source_file_and_url(self):
        pairs = extract_year_month_pairs(
            "treasury_bulletin_2025_03.txt; https://example.test/title/march-2024-123?page=4"
        )

        self.assertIn((2025, 3), pairs)
        self.assertIn((2024, 3), pairs)

    def test_numeric_answer_matching_helper_uses_one_percent_tolerance(self):
        self.assertTrue(answer_matches_gold("The reported amount was $1,009.50.", "$1,000"))
        self.assertFalse(answer_matches_gold("The reported amount was $1,020.", "$1,000"))

    def test_retrieval_metrics_match_relevant_snippets(self):
        chunks = [
            ChunkRecord("a", "Debt was 100 million.", "2024_03.txt", 2024, 3),
            ChunkRecord("b", "Other debt table.", "2024_03.txt", 2024, 3),
            ChunkRecord("c", "Debt was 100 million.", "2024_06.txt", 2024, 6),
        ]
        relevant = relevant_chunk_ids(chunks, {(2024, 3)}, "100")

        self.assertEqual(relevant, {"a"})
        self.assertEqual(first_relevant_rank([chunks[2], chunks[0]], {(2024, 3)}, relevant), 0.5)
        self.assertEqual(safe_divide(1, len(relevant)), 1.0)

    def test_document_metrics_accept_correct_doc_even_when_snippet_differs(self):
        chunks = [
            ChunkRecord("answer", "Debt was 100 million.", "2024_03.txt", 2024, 3),
            ChunkRecord("same_doc", "Different table from same bulletin.", "2024_03.txt", 2024, 3),
            ChunkRecord("other_doc", "Debt was 100 million.", "2024_06.txt", 2024, 6),
        ]
        relevant = relevant_chunk_ids(chunks, {(2024, 3)}, "100")

        self.assertEqual(relevant, {"answer"})
        self.assertEqual(correct_documents_found([chunks[1]], {(2024, 3)}, relevant), 1)
        self.assertEqual(first_relevant_rank([chunks[1], chunks[0]], {(2024, 3)}, relevant), 1.0)
        self.assertEqual(len({"same_doc"} & relevant), 0)

    def test_judge_http_error_fails_evaluation(self):
        with temporary_eval_run() as paths:
            with patch.object(metrics.DeepInfraJudge, "judge", side_effect=metrics.OpenAIError("Bad Request")):
                with self.assertRaises(metrics.OpenAIError):
                    evaluate_predictions(config_path=paths["config_path"])

            self.assertFalse((paths["output_dir"] / "metrics.json").exists())
            self.assertFalse((paths["output_dir"] / "judge_error.json").exists())

    def test_clean_predictions_use_retrieval_log_for_context_ids(self):
        with temporary_eval_run(include_retrieval_ids=False) as paths:
            with (paths["output_dir"] / "retrieval_logs.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"question_id": "Q1", "final_context_ids": ["c1"]}) + "\n")

            with patch.object(
                metrics.DeepInfraJudge,
                "judge",
                return_value=metrics.JudgeResult(
                    supported_claims=1,
                    total_claims=1,
                    fabricated_claims=0,
                    correct_answer=1,
                    raw_response={},
                ),
            ):
                metrics_path = evaluate_predictions(config_path=paths["config_path"])

            written = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(written["hit_rate@5"], 1.0)
            self.assertEqual(written["recall"], 1.0)
            self.assertEqual(written["factual_accuracy"], 1.0)

    def test_factual_accuracy_uses_required_llm_judge(self):
        with temporary_eval_run(predicted_answer="Debt was 999.") as paths:
            with patch.object(
                metrics.DeepInfraJudge,
                "judge",
                return_value=metrics.JudgeResult(
                    supported_claims=1,
                    total_claims=1,
                    fabricated_claims=0,
                    correct_answer=1,
                    raw_response={},
                ),
            ):
                metrics_path = evaluate_predictions(config_path=paths["config_path"])

            written = json.loads(metrics_path.read_text(encoding="utf-8"))
            details = pd.read_csv(paths["output_dir"] / "evaluation_details.csv")
            self.assertEqual(written["factual_accuracy"], 1.0)
            self.assertEqual(details.loc[0, "factual_accuracy"], 1)
            self.assertEqual(details.loc[0, "exact_answer_match"], 0)

    def test_evaluate_predictions_does_not_fallback_on_judge_response_error(self):
        with temporary_eval_run() as paths:
            with patch.object(
                metrics.DeepInfraJudge,
                "judge",
                side_effect=metrics.JudgeResponseError("LLM judge did not return valid JSON"),
            ):
                with self.assertRaises(metrics.JudgeResponseError):
                    evaluate_predictions(config_path=paths["config_path"])

            self.assertFalse((paths["output_dir"] / "judge_error.json").exists())

    def test_deepinfra_judge_requests_json_format(self):
        completion = completion_with_json(
            {
                "supported_claims": 1,
                "total_claims": 1,
                "fabricated_claims": 0,
                "correct_answer": 1,
            },
            model="test-model",
        )
        client = Mock()
        client.chat.completions.create.return_value = completion

        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "test-key"}), patch(
            "common.llm.OpenAI", return_value=client
        ) as openai_class:
            metrics.DeepInfraJudge(
                model="test-model",
                request_sleep_seconds=0,
                retry_sleep_seconds=0,
            ).judge(
                "Question?",
                "Answer.",
                ["Answer."],
            )

        self.assertEqual(openai_class.call_args.kwargs["api_key"], "test-key")
        self.assertEqual(openai_class.call_args.kwargs["base_url"], "https://api.deepinfra.com/v1/openai")
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["temperature"], 0.0)

    def test_deepinfra_judge_accepts_claim_lists(self):
        completion = completion_with_json(
            {
                "supported_claims": ["Claim one", "Claim two"],
                "total_claims": ["Claim one", "Claim two", "Claim three"],
                "fabricated_claims": ["Claim three"],
                "correct_answer": True,
            }
        )
        client = Mock()
        client.chat.completions.create.return_value = completion

        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "test-key"}), patch(
            "common.llm.OpenAI", return_value=client
        ):
            result = metrics.DeepInfraJudge(
                model="test-model",
                request_sleep_seconds=0,
                retry_sleep_seconds=0,
            ).judge(
                "Question?",
                "Answer.",
                ["Answer."],
            )

        self.assertEqual(result.supported_claims, 2)
        self.assertEqual(result.total_claims, 3)
        self.assertEqual(result.fabricated_claims, 1)
        self.assertEqual(result.correct_answer, 1)

    def test_deepinfra_judge_retries_invalid_json_response(self):
        invalid_completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))],
        )
        valid_completion = completion_with_json(
            {
                "supported_claims": 1,
                "total_claims": 1,
                "fabricated_claims": 0,
                "correct_answer": 1,
            },
            model="test-model",
        )
        client = Mock()
        client.chat.completions.create.side_effect = [invalid_completion, valid_completion]

        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "test-key"}), patch(
            "common.llm.OpenAI", return_value=client
        ):
            result = metrics.DeepInfraJudge(
                model="test-model",
                max_retries=3,
                request_sleep_seconds=0,
                retry_sleep_seconds=0,
            ).judge(
                "Question?",
                "Answer.",
                ["Answer."],
            )

        self.assertEqual(result.supported_claims, 1)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_deepinfra_judge_fails_after_invalid_json_retries(self):
        invalid_completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))],
        )
        client = Mock()
        client.chat.completions.create.return_value = invalid_completion

        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "test-key"}), patch(
            "common.llm.OpenAI", return_value=client
        ):
            with self.assertRaises(metrics.JudgeResponseError):
                metrics.DeepInfraJudge(
                    model="test-model",
                    max_retries=3,
                    request_sleep_seconds=0,
                    retry_sleep_seconds=0,
                ).judge(
                    "Question?",
                    "Answer.",
                    ["Answer."],
                )

        self.assertEqual(client.chat.completions.create.call_count, 4)

    def test_deepinfra_judge_reads_reasoning_content_when_message_content_is_empty(self):
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        reasoning_content=json.dumps(
                            {
                                "supported_claims": 2,
                                "total_claims": 2,
                                "fabricated_claims": 0,
                                "correct_answer": "yes",
                            }
                        ),
                    )
                )
            ],
        )
        client = Mock()
        client.chat.completions.create.return_value = completion

        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "test-key"}), patch(
            "common.llm.OpenAI", return_value=client
        ):
            result = metrics.DeepInfraJudge(
                model="test-model",
                request_sleep_seconds=0,
                retry_sleep_seconds=0,
            ).judge(
                "Question?",
                "Answer.",
                ["Answer."],
            )

        self.assertEqual(result.supported_claims, 2)
        self.assertEqual(result.total_claims, 2)
        self.assertEqual(result.fabricated_claims, 0)
        self.assertEqual(result.correct_answer, 1)

    def test_judge_completion_message_content_reads_model_dump_text_fields(self):
        class DumpOnlyCompletion:
            def __init__(self):
                self.choices = [SimpleNamespace(message=SimpleNamespace(content=""))]

            def model_dump(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning": (
                                    "Final answer: "
                                    '{"supported_claims": 1, "total_claims": 2, '
                                    '"fabricated_claims": 1, "correct_answer": 0}'
                                ),
                            }
                        }
                    ]
                }

        text = metrics.judge_completion_message_content(DumpOnlyCompletion())

        self.assertIn('"supported_claims": 1', text)

    def test_judge_count_unwraps_single_item_numeric_list(self):
        self.assertEqual(judge_count(["2"], "supported_claims"), 2)

    def test_judge_bool_accepts_common_judge_shapes(self):
        self.assertEqual(judge_bool(True, "correct_answer"), 1)
        self.assertEqual(judge_bool("yes", "correct_answer"), 1)
        self.assertEqual(judge_bool({"value": 0}, "correct_answer"), 0)


def temporary_eval_run(predicted_answer="Debt was 100.", include_retrieval_ids=True):
    return TemporaryEvalRun(predicted_answer, include_retrieval_ids)


class TemporaryEvalRun:
    def __init__(self, predicted_answer: str, include_retrieval_ids: bool) -> None:
        self.predicted_answer = predicted_answer
        self.include_retrieval_ids = include_retrieval_ids
        self.tempdir: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> dict[str, Path]:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        output_dir = root / "outputs"
        output_dir.mkdir()
        csv_path = root / "officeqa.csv"
        config_path = root / "config.yaml"

        pd.DataFrame(
            [
                {
                    "uid": "Q1",
                    "question": "What was debt in March 2024?",
                    "answer": "100",
                    "source_files": "treasury_bulletin_2024_03.txt",
                }
            ]
        ).to_csv(csv_path, index=False)

        prediction_row = {
            "question_id": "Q1",
            "question": "What was debt in March 2024?",
            "gold_answer": "100",
            "predicted_answer": self.predicted_answer,
            "detected_year": 2024,
            "detected_month": 3,
            "retrieval_method": "test",
        }
        if self.include_retrieval_ids:
            prediction_row["retrieved_context_ids"] = json.dumps(["c1"])
        pd.DataFrame([prediction_row]).to_csv(output_dir / "predictions.csv", index=False)

        with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "text": "Debt was 100.",
                        "metadata": {
                            "chunk_id": "c1",
                            "source_path": "treasury_bulletin_2024_03.txt",
                            "year": 2024,
                            "month": 3,
                        },
                    }
                )
                + "\n"
            )

        config_path.write_text(f"csv_path: {csv_path}\noutput_dir: {output_dir}\n", encoding="utf-8")
        return {"root": root, "output_dir": output_dir, "csv_path": csv_path, "config_path": config_path}

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.tempdir is not None:
            self.tempdir.cleanup()


def completion_with_json(payload: dict, model: str | None = None):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
    )


if __name__ == "__main__":
    unittest.main()
