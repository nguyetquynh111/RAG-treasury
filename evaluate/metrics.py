"""Evaluate Treasury RAG predictions with retrieval and answer metrics."""

from __future__ import annotations

import argparse
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from common.chunking import load_chunks as load_chunk_records
from common.config import DEFAULT_CONFIG_PATH, load_pipeline_config, load_yaml_config, require_keys, resolve_path
from common.dataset import YEAR_PATTERN, extract_all_year_month
from common.llm import (
    DEFAULT_DEEPINFRA_API_KEY_ENV,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_REQUEST_SLEEP_SECONDS,
    DEFAULT_RETRY_SLEEP_SECONDS,
    GenerationSettings,
    chat_completion_message_content,
    create_chat_completion_with_rate_limit_sleep,
    load_env_file,
    OpenAIError,
    sleep_if_positive,
)


# DEFAULT_CONFIG_PATH is imported from common.config and points to config/config.yaml
DEFAULT_JUDGE_MODEL = DEFAULT_GENERATION_MODEL
METRIC_KEYS = [
    "hit_rate@5",
    "mrr",
    "recall",
    "groundedness",
    "factual_accuracy",
    "hallucination_rate",
]
NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    source_path: str
    year: int
    month: int


@dataclass(frozen=True)
class JudgeResult:
    supported_claims: int
    total_claims: int
    fabricated_claims: int
    correct_answer: int
    raw_response: dict[str, Any]


class JudgeResponseError(ValueError):
    """Raised when the judge response cannot be parsed or validated."""


class DeepInfraJudge:
    """Small OpenAI-compatible judge client for answer-metric scoring."""

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key_env: str = DEFAULT_DEEPINFRA_API_KEY_ENV,
        timeout_seconds: int = 90,
        max_tokens: int = 512,
        request_sleep_seconds: float = DEFAULT_REQUEST_SLEEP_SECONDS,
        retry_sleep_seconds: float = DEFAULT_RETRY_SLEEP_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.request_sleep_seconds = request_sleep_seconds
        self.retry_sleep_seconds = retry_sleep_seconds
        self.max_retries = max_retries

    def judge(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        gold_answer: str = "",
    ) -> JudgeResult:
        prompt = build_judge_prompt(question, answer, contexts, gold_answer)
        for attempt in range(self.max_retries + 1):
            completion = create_chat_completion_with_rate_limit_sleep(
                settings=self._settings(),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            message_content = judge_completion_message_content(completion)
            try:
                parsed = parse_json_object(message_content)
                supported_claims = judge_count(parsed.get("supported_claims", 0), "supported_claims")
                total_claims = judge_count(parsed.get("total_claims", 0), "total_claims")
                fabricated_claims = judge_count(parsed.get("fabricated_claims", 0), "fabricated_claims")
                correct_answer = judge_bool(parsed.get("correct_answer", False), "correct_answer")
            except ValueError as exc:
                if attempt == self.max_retries:
                    raise JudgeResponseError(
                        f"LLM judge did not return valid JSON after {self.max_retries} retries: {exc}"
                    ) from exc
                sleep_if_positive(self.retry_sleep_seconds)
                continue

            return JudgeResult(
                supported_claims=supported_claims,
                total_claims=total_claims,
                fabricated_claims=fabricated_claims,
                correct_answer=correct_answer,
                raw_response={
                    "message": {"content": message_content},
                    "parsed": parsed,
                    "model": getattr(completion, "model", self.model),
                },
            )

        raise RuntimeError("Unexpected judge response retry loop exit.")

    def _settings(self) -> GenerationSettings:
        """Return settings compatible with the shared OpenAI client helper."""
        return GenerationSettings(
            backend="deepinfra",
            model=self.model,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            timeout_seconds=self.timeout_seconds,
            max_tokens=self.max_tokens,
            temperature=0.0,
            max_context_chars=0,
            allow_extractive_fallback=False,
            require_citations=False,
            fallback_on_ungrounded_answer=False,
            request_sleep_seconds=self.request_sleep_seconds,
            retry_sleep_seconds=self.retry_sleep_seconds,
            max_retries=self.max_retries,
            keep_alive="",
            num_ctx=None,
        )

def evaluate_predictions(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    predictions_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    index_dir: str | Path | None = None,
    judge_model: str | None = None,
    mode: str = "engineered",
) -> Path:
    """Evaluate predictions and write metrics.json with exactly six metrics at K=5.

    Prediction artifacts are mode-specific. Chunk artifacts are read from the
    shared index directory so baseline and engineered runs are evaluated against
    the exact same chunk database.
    """
    config_path = Path(config_path)
    config = load_eval_config(config_path, mode=mode)
    resolved_output_dir = (
        Path(output_dir) if output_dir else resolve_path(config_path, config["output_dir"])
    )
    resolved_index_dir = (
        Path(index_dir)
        if index_dir
        else resolve_path(config_path, config.get("index_dir", config["output_dir"]))
    )
    resolved_predictions_path = (
        Path(predictions_path)
        if predictions_path
        else resolved_output_dir / "predictions.csv"
    )
    chunks_path = resolved_index_dir / "chunks.jsonl"
    csv_path = resolve_path(config_path, config["csv_path"])

    predictions = load_predictions(resolved_predictions_path)
    retrieval_logs = load_retrieval_logs(resolved_output_dir / "retrieval_logs.jsonl")
    chunks = load_chunks(chunks_path)
    answer_key = pd.read_csv(csv_path)
    answer_lookup = build_answer_lookup(answer_key)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    load_env_file()
    generation_config = config.get("generation", {})
    judge = DeepInfraJudge(
        model=judge_model or str(generation_config.get("model", DEFAULT_JUDGE_MODEL)),
        base_url=str(generation_config.get("base_url", DEFAULT_OPENAI_BASE_URL)),
        api_key_env=str(generation_config.get("api_key_env", DEFAULT_DEEPINFRA_API_KEY_ENV)),
        timeout_seconds=int(generation_config.get("timeout_seconds", 90)),
        max_tokens=int(generation_config.get("max_tokens", 512)),
        request_sleep_seconds=float(
            generation_config.get("request_sleep_seconds", DEFAULT_REQUEST_SLEEP_SECONDS)
        ),
        retry_sleep_seconds=float(
            generation_config.get("retry_sleep_seconds", DEFAULT_RETRY_SLEEP_SECONDS)
        ),
        max_retries=int(generation_config.get("max_retries", DEFAULT_MAX_RETRIES)),
    )

    detail_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    hit_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    total_relevant_snippets = 0
    total_found_relevant_snippets = 0
    correct_answers = 0
    supported_claims = 0
    total_claims = 0
    fabricated_claims = 0

    for _, row in predictions.iterrows():
        row_data = row.to_dict()
        question_id = str(row_data.get("question_id", row_data.get("row_id", "")))
        row_data = merge_retrieval_log(row_data, retrieval_logs.get(question_id))
        answer_row = lookup_answer_row(row_data, answer_lookup, answer_key)
        question = str(row_data.get("question", answer_row.get("question", "")))
        gold_answer = str(row_data.get("gold_answer", answer_row.get("answer", "")))
        predicted_answer = prediction_text(row_data)

        retrieved_ids = retrieved_chunk_ids(row_data)
        retrieved_top_5_ids = retrieved_ids[:5]
        retrieved_chunks = [chunks_by_id[chunk_id] for chunk_id in retrieved_ids if chunk_id in chunks_by_id]
        retrieved_top_5_chunks = retrieved_chunks[:5]
        relevant_doc_pairs = relevant_source_pairs(answer_row, row_data)
        relevant_snippet_ids = relevant_chunk_ids(chunks, relevant_doc_pairs, gold_answer)
        correct_docs_found = correct_documents_found(
            retrieved_top_5_chunks,
            relevant_doc_pairs,
            relevant_snippet_ids,
        )

        hit = 1.0 if correct_docs_found else 0.0
        reciprocal_rank = first_relevant_rank(retrieved_top_5_chunks, relevant_doc_pairs, relevant_snippet_ids)
        found_snippets = len(set(retrieved_top_5_ids) & relevant_snippet_ids)
        relevant_count = len(relevant_snippet_ids)
        exact_answer_match = int(answer_matches_gold(predicted_answer, gold_answer))

        hit_scores.append(hit)
        reciprocal_ranks.append(reciprocal_rank)
        total_found_relevant_snippets += found_snippets
        total_relevant_snippets += relevant_count

        contexts = [chunk.text for chunk in retrieved_chunks[:5]]
        judge_result = judge.judge(question, predicted_answer, contexts, gold_answer)
        correct_answers += judge_result.correct_answer
        supported_claims += judge_result.supported_claims
        total_claims += judge_result.total_claims
        fabricated_claims += judge_result.fabricated_claims
        judge_rows.append(
            {
                "question_id": question_id,
                "supported_claims": judge_result.supported_claims,
                "total_claims": judge_result.total_claims,
                "fabricated_claims": judge_result.fabricated_claims,
                "judge_correct_answer": judge_result.correct_answer,
                "raw_response": judge_result.raw_response,
            }
        )

        detail_rows.append(
            {
                "question_id": question_id,
                "hit_rate@5": hit,
                "correct_docs_found@5": correct_docs_found,
                "reciprocal_rank": reciprocal_rank,
                "relevant_snippets_found@5": found_snippets,
                "total_relevant_snippets": relevant_count,
                "factual_accuracy": judge_result.correct_answer,
                "exact_answer_match": exact_answer_match,
                "judge_correct_answer": judge_result.correct_answer,
                "supported_claims": judge_result.supported_claims,
                "total_claims": judge_result.total_claims,
                "fabricated_claims": judge_result.fabricated_claims,
            }
        )

    metrics = {
        "hit_rate@5": mean(hit_scores),
        "mrr": mean(reciprocal_ranks),
        "recall": safe_divide(total_found_relevant_snippets, total_relevant_snippets),
        "groundedness": safe_divide(supported_claims, total_claims, default=1.0),
        "factual_accuracy": safe_divide(correct_answers, len(predictions)),
        "hallucination_rate": safe_divide(fabricated_claims, total_claims),
    }
    metrics = {key: metrics[key] for key in METRIC_KEYS}

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = resolved_output_dir / "metrics.json"
    judge_error_path = resolved_output_dir / "judge_error.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(detail_rows).to_csv(resolved_output_dir / "evaluation_details.csv", index=False)
    with (resolved_output_dir / "judge_results.jsonl").open("w", encoding="utf-8") as handle:
        for judge_row in judge_rows:
            handle.write(json.dumps(judge_row) + "\n")
    if judge_error_path.exists():
        judge_error_path.unlink()
    return metrics_path


def load_eval_config(config_path: Path, *, mode: str = "engineered") -> dict[str, Any]:
    """Load evaluation config from the shared config file or a tiny test config."""
    raw_config = load_yaml_config(config_path, label="Evaluation")
    if "runs" in raw_config:
        return load_pipeline_config(config_path, mode=mode)
    require_keys(raw_config, ("csv_path", "output_dir"), path=config_path)
    raw_config.setdefault("index_dir", raw_config["output_dir"])
    return raw_config


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_path}. Run QA first.")
    predictions = pd.read_csv(predictions_path)
    if predictions.empty:
        raise ValueError(f"Predictions CSV is empty: {predictions_path}")
    return predictions


def load_retrieval_logs(logs_path: Path) -> dict[str, dict[str, Any]]:
    """Load optional JSONL retrieval logs keyed by question id."""
    if not logs_path.exists():
        return {}

    logs: dict[str, dict[str, Any]] = {}
    with logs_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in retrieval log {logs_path}:{line_number}") from exc
            question_id = record.get("question_id")
            if question_id is not None and not is_missing(question_id):
                logs[str(question_id)] = record
    return logs


def merge_retrieval_log(row_data: dict[str, Any], log_row: dict[str, Any] | None) -> dict[str, Any]:
    """Combine clean prediction rows with optional sidecar retrieval metadata."""
    if not log_row:
        return row_data
    merged = dict(log_row)
    merged.update(row_data)
    return merged


def load_chunks(chunks_path: Path) -> list[ChunkRecord]:
    """Load JSONL chunks into evaluation records."""
    records = load_chunk_records(
        chunks_path,
        required_metadata_keys=("chunk_id", "source_path", "year", "month"),
        not_found_hint="Build the retrieval index first.",
    )
    return [
        ChunkRecord(
            chunk_id=str(record["metadata"]["chunk_id"]),
            text=str(record["text"]),
            source_path=str(record["metadata"]["source_path"]),
            year=int(record["metadata"]["year"]),
            month=int(record["metadata"]["month"]),
        )
        for record in records
    ]


def build_answer_lookup(answer_key: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if "uid" in answer_key.columns:
        for _, row in answer_key.iterrows():
            lookup[str(row["uid"])] = row.to_dict()
    return lookup


def lookup_answer_row(
    prediction_row: dict[str, Any],
    answer_lookup: dict[str, dict[str, Any]],
    answer_key: pd.DataFrame,
) -> dict[str, Any]:
    question_id = prediction_row.get("question_id")
    if question_id is not None and not is_missing(question_id):
        match = answer_lookup.get(str(question_id))
        if match is not None:
            return match

    row_id = prediction_row.get("row_id")
    if row_id is not None and not is_missing(row_id):
        index = int(row_id)
        if 0 <= index < len(answer_key):
            return answer_key.iloc[index].to_dict()

    question = str(prediction_row.get("question", ""))
    if question and "question" in answer_key.columns:
        matches = answer_key[answer_key["question"].astype(str) == question]
        if not matches.empty:
            return matches.iloc[0].to_dict()
    return {}


def prediction_text(row: dict[str, Any]) -> str:
    for column in ["predicted_answer", "prediction", "answer"]:
        value = row.get(column)
        if value is not None and not is_missing(value):
            return str(value)
    return ""


def retrieved_chunk_ids(row: dict[str, Any]) -> list[str]:
    for column in ["retrieved_context_ids", "final_context_ids"]:
        context_ids = parse_json_value(row.get(column))
        if isinstance(context_ids, list):
            return [str(chunk_id) for chunk_id in context_ids]

    for column in ["retrieved_context", "final_context"]:
        contexts = parse_json_value(row.get(column))
        if isinstance(contexts, list):
            ids = []
            for item in contexts:
                if not isinstance(item, dict):
                    continue
                metadata = item.get("metadata", {})
                chunk_id = item.get("chunk_id") or metadata.get("chunk_id")
                if chunk_id is not None:
                    ids.append(str(chunk_id))
            return ids

    for column in ["retrieved_sources", "final_sources"]:
        sources = parse_json_value(row.get(column))
        if isinstance(sources, list):
            ids = []
            for item in sources:
                if isinstance(item, dict) and item.get("chunk_id") is not None:
                    ids.append(str(item["chunk_id"]))
            return ids

    return []


def relevant_source_pairs(answer_row: dict[str, Any], prediction_row: dict[str, Any]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for column in ["source_files", "source_docs", "source_file", "source_doc", "date", "period"]:
        value = answer_row.get(column)
        if value is not None and not is_missing(value):
            pairs.update(extract_year_month_pairs(str(value)))

    if not pairs:
        for year_key, month_key in [("year", "month"), ("detected_year", "detected_month")]:
            year = prediction_row.get(year_key)
            month = prediction_row.get(month_key)
            if year is not None and month is not None and not is_missing(year) and not is_missing(month):
                pairs.add((int(year), int(float(month))))
    return pairs


def relevant_chunk_ids(
    chunks: list[ChunkRecord],
    source_pairs: set[tuple[int, int]],
    gold_answer: str,
) -> set[str]:
    source_chunks = {
        chunk.chunk_id
        for chunk in chunks
        if not source_pairs or (chunk.year, chunk.month) in source_pairs
    }
    containing_answer = {
        chunk.chunk_id
        for chunk in chunks
        if chunk.chunk_id in source_chunks and answer_appears_in_text(gold_answer, chunk.text)
    }
    return containing_answer or source_chunks


def is_relevant_chunk(
    chunk: ChunkRecord,
    source_pairs: set[tuple[int, int]],
    relevant_snippet_ids: set[str],
) -> bool:
    """Return whether a chunk is in a correct source document for Hit@5/MRR."""
    return is_relevant_document_chunk(chunk, source_pairs, relevant_snippet_ids)


def is_relevant_document_chunk(
    chunk: ChunkRecord,
    source_pairs: set[tuple[int, int]],
    relevant_snippet_ids: set[str],
) -> bool:
    """Return whether a chunk belongs to a correct document.

    Hit Rate@5 and MRR judge whether search found the right page/document. When
    source metadata is available, the whole source document is correct for those
    metrics. If metadata is missing, fall back to exact relevant snippets.
    """
    if source_pairs:
        return (chunk.year, chunk.month) in source_pairs
    if chunk.chunk_id in relevant_snippet_ids:
        return True
    return False


def correct_documents_found(
    retrieved_chunks: list[ChunkRecord],
    source_pairs: set[tuple[int, int]],
    relevant_snippet_ids: set[str],
) -> int:
    """Count distinct correct documents represented in the retrieved top-k list."""
    if source_pairs:
        return len(
            {
                (chunk.year, chunk.month)
                for chunk in retrieved_chunks
                if (chunk.year, chunk.month) in source_pairs
            }
        )
    return len(
        {chunk.chunk_id for chunk in retrieved_chunks if chunk.chunk_id in relevant_snippet_ids}
    )


def first_relevant_rank(
    retrieved_chunks: list[ChunkRecord],
    source_pairs: set[tuple[int, int]],
    relevant_snippet_ids: set[str],
) -> float:
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if is_relevant_document_chunk(chunk, source_pairs, relevant_snippet_ids):
            return 1.0 / rank
    return 0.0


def answer_matches_gold(prediction: str, gold_answer: str, tolerance: float = 0.01) -> bool:
    gold_numbers = extract_numbers(gold_answer)
    prediction_numbers = extract_numbers(prediction)
    if gold_numbers:
        if not prediction_numbers:
            return False
        for gold_number in gold_numbers:
            if not any(numbers_close(candidate, gold_number, tolerance) for candidate in prediction_numbers):
                return False
        return True
    return normalize_answer(prediction) == normalize_answer(gold_answer)


def answer_appears_in_text(answer: str, text: str) -> bool:
    normalized_answer = normalize_answer(answer)
    normalized_text = normalize_answer(text)
    if normalized_answer and normalized_answer in normalized_text:
        return True
    answer_numbers = extract_numbers(answer)
    if not answer_numbers:
        return False
    text_numbers = extract_numbers(text)
    return all(
        any(numbers_close(candidate, target, tolerance=0.001) for candidate in text_numbers)
        for target in answer_numbers
    )


def extract_numbers(text: str) -> list[float]:
    numbers: list[float] = []
    for match in NUMBER_PATTERN.finditer(str(text)):
        raw = match.group().replace("$", "").replace(",", "").replace("%", "")
        try:
            numbers.append(float(raw))
        except ValueError:
            continue
    return numbers


def numbers_close(candidate: float, target: float, tolerance: float) -> bool:
    if target == 0:
        return abs(candidate) <= tolerance
    return abs(candidate - target) / abs(target) <= tolerance


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def extract_year_month_pairs(value: str) -> set[tuple[int, int]]:
    """Extract source year/month pairs with the shared date parser."""
    return set(extract_all_year_month(value))


def build_judge_prompt(
    question: str,
    answer: str,
    contexts: list[str],
    gold_answer: str = "",
) -> str:
    clipped_contexts = []
    for index, context in enumerate(contexts, start=1):
        clipped = " ".join(str(context).split())
        clipped_contexts.append(f"[Source {index}] {clipped[:4000]}")
    context_block = "\n\n".join(clipped_contexts) if clipped_contexts else "[No retrieved source text]"
    return (
        "You are evaluating a RAG answer against retrieved Treasury Bulletin sources.\n"
        "Break the answer into atomic factual claims. A claim is supported only if the sources directly support it.\n"
        "A fabricated claim is any factual claim that is not supported by the sources or contradicts them.\n"
        "Also compare the answer with the CSV gold answer for the question. Set correct_answer to 1 only if "
        "the generated answer gives the same final answer as the gold answer; otherwise set it to 0. "
        "For numeric answers, allow normal formatting differences and values within plus or minus 1 percent.\n"
        "Return only valid JSON with these keys: supported_claims, total_claims, fabricated_claims, "
        "correct_answer, rationale.\n\n"
        f"Question:\n{question}\n\n"
        f"CSV gold answer:\n{gold_answer}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Retrieved sources:\n{context_block}\n"
    )


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"LLM judge did not return JSON: {text[:300]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM judge JSON response must be an object.")
    return parsed


def judge_completion_message_content(completion: Any) -> str:
    """Extract judge JSON from OpenAI-compatible chat responses, including reasoning fields."""
    candidates = completion_text_candidates(completion)
    for candidate in candidates:
        if re.search(r"\{.*\}", candidate, flags=re.DOTALL):
            return candidate
    return candidates[0] if candidates else ""


def completion_text_candidates(completion: Any) -> list[str]:
    candidates: list[str] = []
    try:
        add_text_candidate(candidates, chat_completion_message_content(completion))
    except (AttributeError, IndexError, KeyError, TypeError):
        pass

    for choice in as_list(get_value(completion, "choices")):
        for container_name in ["message", "delta"]:
            container = get_value(choice, container_name)
            for field_name in ["content", "reasoning_content", "reasoning", "text"]:
                add_text_candidate(candidates, get_value(container, field_name))
        add_text_candidate(candidates, get_value(choice, "text"))

    for field_name in ["output_text", "content", "text", "reasoning_content", "reasoning"]:
        add_text_candidate(candidates, get_value(completion, field_name))

    dump = dump_response_object(completion)
    if dump is not None:
        collect_response_text(dump, candidates)

    return candidates


def collect_response_text(value: Any, candidates: list[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"content", "text", "output_text", "reasoning", "reasoning_content"}:
                add_text_candidate(candidates, nested)
            else:
                collect_response_text(nested, candidates)
    elif isinstance(value, list):
        for item in value:
            collect_response_text(item, candidates)


def add_text_candidate(candidates: list[str], value: Any) -> None:
    text = stringify_response_text(value)
    if text and text not in candidates:
        candidates.append(text)


def stringify_response_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(stringify_response_text(item))
        return "".join(parts).strip()
    if isinstance(value, dict):
        for key in ["text", "content", "output_text", "reasoning_content", "reasoning"]:
            if key in value:
                return stringify_response_text(value[key])
        return ""
    return str(value).strip()


def get_value(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def dump_response_object(value: Any) -> Any:
    for method_name in ["model_dump", "dict"]:
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return method()
            except TypeError:
                continue
    return value if isinstance(value, (dict, list)) else None


def judge_count(value: Any, field_name: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 0
        try:
            return max(0, int(float(stripped)))
        except ValueError as exc:
            raise ValueError(f"LLM judge field {field_name!r} must be numeric, got {value!r}") from exc
    if isinstance(value, list):
        if len(value) == 1 and is_numeric_judge_scalar(value[0]):
            return judge_count(value[0], field_name)
        return len(value)
    if isinstance(value, dict):
        for key in ["count", "value", "total"]:
            if key in value:
                return judge_count(value[key], field_name)
        for key in ["claims", "items"]:
            nested = value.get(key)
            if isinstance(nested, list):
                return len(nested)
    raise ValueError(f"LLM judge field {field_name!r} must be numeric, got {type(value).__name__}")


def judge_bool(value: Any, field_name: str) -> int:
    """Parse an LLM judge boolean/count field as 0 or 1."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"true", "yes", "y", "correct", "1"}:
            return 1
        if stripped in {"false", "no", "n", "incorrect", "0", ""}:
            return 0
        try:
            return 1 if float(stripped) >= 0.5 else 0
        except ValueError as exc:
            raise ValueError(f"LLM judge field {field_name!r} must be boolean, got {value!r}") from exc
    if isinstance(value, list):
        if len(value) == 1:
            return judge_bool(value[0], field_name)
        raise ValueError(f"LLM judge field {field_name!r} must be boolean, got list")
    if isinstance(value, dict):
        for key in ["correct_answer", "is_correct", "correct", "count", "value", "total"]:
            if key in value:
                return judge_bool(value[key], field_name)
    raise ValueError(f"LLM judge field {field_name!r} must be boolean, got {type(value).__name__}")


def is_numeric_judge_scalar(value: Any) -> bool:
    if isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.strip())
        except ValueError:
            return False
        return True
    return False


def describe_judge_error(exc: Exception, model: str, question_id: str) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    error: dict[str, Any] = {
        "question_id": question_id,
        "model": model,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    if response is not None:
        error["status_code"] = response.status_code
        error["response_text"] = response.text[:1000]
    return error


def parse_json_value(value: Any) -> Any:
    if value is None or is_missing(value):
        return None
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return None



def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, int)) else False


def safe_divide(numerator: int | float, denominator: int | float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser(description="Evaluate Treasury RAG predictions with six metrics.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to shared pipeline YAML config.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "engineered"],
        default="engineered",
        help="Which run output to evaluate from the shared config.",
    )
    parser.add_argument("--predictions", default=None, help="Optional predictions.csv path.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional run output directory containing predictions/logs.",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Optional shared index directory containing chunks.jsonl.",
    )
    args = parser.parse_args()

    metrics_path = evaluate_predictions(
        config_path=args.config,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        index_dir=args.index_dir,
        mode=args.mode,
    )
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
