"""Shared grounded generation utilities for Treasury RAG pipelines."""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

try:  # pragma: no cover - normal path when requirements are installed
    from openai import OpenAI, OpenAIError, RateLimitError
except ModuleNotFoundError:  # pragma: no cover - lightweight test environments
    class OpenAIError(Exception):
        """Fallback OpenAI error used when the openai package is not installed."""

    class RateLimitError(OpenAIError):
        """Fallback rate-limit error used when the openai package is not installed."""

    class OpenAI:  # type: ignore[no-redef]
        """Placeholder client that fails only if an API call is attempted."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("Install openai>=1.0.0 to call the LLM API.")


DEFAULT_OPENAI_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_DEEPINFRA_API_KEY_ENV = "DEEPINFRA_API_KEY"
DEFAULT_GENERATION_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning"
SUPPORTED_GENERATION_BACKENDS = {"deepinfra", "openai", "extractive"}
SOURCE_CITATION_PATTERN = re.compile(r"\[S\d+\]")
DEFAULT_REQUEST_SLEEP_SECONDS = 2.0
DEFAULT_RETRY_SLEEP_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 3


@dataclass(frozen=True)
class SourceSnippet:
    """One bounded retrieved source passed to the answer generator."""

    label: str
    text: str
    citation: str


@dataclass(frozen=True)
class GenerationSettings:
    """Validated answer-generation settings loaded from YAML config."""

    backend: str
    model: str
    base_url: str
    api_key_env: str
    timeout_seconds: int
    max_tokens: int
    temperature: float
    max_context_chars: int
    allow_extractive_fallback: bool
    require_citations: bool
    fallback_on_ungrounded_answer: bool
    request_sleep_seconds: float
    retry_sleep_seconds: float
    max_retries: int
    keep_alive: str
    num_ctx: int | None

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        default_max_context_chars: int,
    ) -> "GenerationSettings":
        """Create generation settings from the pipeline config dictionary."""
        generation_config = config.get("generation", {})
        backend = str(generation_config.get("backend", "deepinfra")).lower()
        if backend not in SUPPORTED_GENERATION_BACKENDS:
            allowed = ", ".join(sorted(SUPPORTED_GENERATION_BACKENDS))
            raise ValueError(f"generation.backend must be one of: {allowed}.")

        return cls(
            backend=backend,
            model=str(generation_config.get("model", DEFAULT_GENERATION_MODEL)),
            base_url=str(generation_config.get("base_url", DEFAULT_OPENAI_BASE_URL)),
            api_key_env=str(generation_config.get("api_key_env", DEFAULT_DEEPINFRA_API_KEY_ENV)),
            timeout_seconds=int(generation_config.get("timeout_seconds", 90)),
            max_tokens=int(generation_config.get("max_tokens", 512)),
            temperature=float(generation_config.get("temperature", 0.0)),
            max_context_chars=int(generation_config.get("max_context_chars", default_max_context_chars)),
            allow_extractive_fallback=bool(generation_config.get("allow_extractive_fallback", False)),
            require_citations=bool(generation_config.get("require_citations", True)),
            fallback_on_ungrounded_answer=bool(
                generation_config.get("fallback_on_ungrounded_answer", True)
            ),
            request_sleep_seconds=float(
                generation_config.get("request_sleep_seconds", DEFAULT_REQUEST_SLEEP_SECONDS)
            ),
            retry_sleep_seconds=float(
                generation_config.get("retry_sleep_seconds", DEFAULT_RETRY_SLEEP_SECONDS)
            ),
            max_retries=int(generation_config.get("max_retries", DEFAULT_MAX_RETRIES)),
            keep_alive=str(generation_config.get("keep_alive", "30m")),
            num_ctx=optional_positive_int(generation_config.get("num_ctx")),
        )


ExtractiveFallback = Callable[[str, list[Any]], str]
SnippetBuilder = Callable[[list[Any], int], list[SourceSnippet]]


class GroundedRAGAnswerGenerator:
    """Generate answers only from retrieved RAG context."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        extractive_fallback: ExtractiveFallback,
        source_snippet_builder: SnippetBuilder,
        default_max_context_chars: int,
    ) -> None:
        self.settings = GenerationSettings.from_config(
            config,
            default_max_context_chars=default_max_context_chars,
        )
        self.extractive_fallback = extractive_fallback
        self.source_snippet_builder = source_snippet_builder

    @property
    def actual_backend(self) -> str:
        """Return the effective answer backend name for logs."""
        return self.settings.backend

    @property
    def actual_model(self) -> str:
        """Return the effective answer model name for logs."""
        if self.settings.backend in {"deepinfra", "openai"}:
            return self.settings.model
        return "extractive"

    def generate(self, question: str, retrieved_items: Sequence[Any]) -> str:
        """Return a cited answer grounded in retrieved context, or NOT_FOUND."""
        items = list(retrieved_items)
        if not items:
            return "NOT_FOUND"

        if self.settings.backend == "extractive":
            return self.extractive_fallback(question, items)

        load_env_file()
        try:
            answer = self._generate_with_openai_compatible_api(question, items)
            return self._guard_grounded_answer(answer, question, items)
        except (OpenAIError, KeyError, IndexError, TypeError, ValueError) as exc:
            if self.settings.allow_extractive_fallback:
                print(
                    f"[generation] LLM failed ({type(exc).__name__}); using extractive fallback.",
                    file=sys.stderr,
                    flush=True,
                )
                return self.extractive_fallback(question, items)
            raise

    def _generate_with_openai_compatible_api(
        self,
        question: str,
        retrieved_items: list[Any],
    ) -> str:
        """Call OpenAI-compatible chat completions with bounded retrieved context."""
        snippets = self.source_snippet_builder(
            retrieved_items,
            self.settings.max_context_chars,
        )
        completion = create_chat_completion_with_rate_limit_sleep(
            settings=self.settings,
            messages=build_messages(question, snippets),
        )
        answer = chat_completion_message_content(completion).strip()
        return answer or "NOT_FOUND"

    def _guard_grounded_answer(
        self,
        answer: str,
        question: str,
        retrieved_items: list[Any],
    ) -> str:
        """Reject empty, NOT_FOUND, or uncited answers before writing predictions."""
        normalized = " ".join(str(answer).split())
        if not normalized:
            return "NOT_FOUND"
        if normalized.upper().startswith("NOT_FOUND"):
            return "NOT_FOUND"
        if self.settings.require_citations and not has_source_citation(normalized):
            if self.settings.fallback_on_ungrounded_answer:
                return self.extractive_fallback(question, retrieved_items)
            return "NOT_FOUND"
        return normalized


def sleep_if_positive(seconds: float) -> None:
    """Sleep for a non-negative number of seconds before or between API calls."""
    if seconds > 0:
        time.sleep(seconds)


def optional_positive_int(value: Any) -> int | None:
    """Return a positive integer when configured, otherwise None."""
    if value is None:
        return None
    integer = int(value)
    if integer <= 0:
        raise ValueError("generation.num_ctx must be positive when provided.")
    return integer


def resolve_api_key(api_key_env: str = DEFAULT_DEEPINFRA_API_KEY_ENV) -> str:
    """Return the configured API key from the environment."""
    api_key = os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"Missing API key. Set {api_key_env} in .env or the environment.")
    return api_key


def create_openai_client(settings: GenerationSettings) -> OpenAI:
    """Create an OpenAI-compatible client for the configured endpoint."""
    return OpenAI(
        api_key=resolve_api_key(settings.api_key_env),
        base_url=settings.base_url,
    )


def create_chat_completion_with_rate_limit_sleep(
    *,
    settings: GenerationSettings,
    messages: list[dict[str, str]],
    response_format: dict[str, str] | None = None,
) -> Any:
    """Create a chat completion with configurable sleep and simple 429 retries."""
    client = create_openai_client(settings)
    for attempt in range(settings.max_retries + 1):
        sleep_if_positive(settings.request_sleep_seconds)
        try:
            kwargs: dict[str, Any] = {
                "model": settings.model,
                "messages": messages,
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
                "timeout": settings.timeout_seconds,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            return client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            if attempt == settings.max_retries:
                raise
            sleep_if_positive(rate_limit_sleep_seconds(getattr(exc, "response", None), settings.retry_sleep_seconds))

    raise RuntimeError("Unexpected chat completion retry loop exit.")


def chat_completion_message_content(completion: Any) -> str:
    """Extract content from an OpenAI-compatible chat completion object."""
    content = completion.choices[0].message.content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def rate_limit_sleep_seconds(response: Any, fallback_seconds: float) -> float:
    """Return retry sleep seconds, preferring Retry-After when present."""
    if response is None:
        return fallback_seconds
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        return fallback_seconds
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        return fallback_seconds


def build_messages(question: str, snippets: Sequence[SourceSnippet]) -> list[dict[str, str]]:
    """Build the strict grounded RAG prompt sent to the chat model."""
    context_block = format_context_block(snippets)
    system_prompt = (
        "You are a Treasury Bulletin RAG assistant. Answer only from the retrieved context. "
        "Do not use outside knowledge, memory, or guesses. If the retrieved context does not "
        "directly contain enough evidence, return exactly NOT_FOUND. Prefer a short answer. "
        "For numeric financial questions, quote the exact value from the source. If calculation "
        "is required, show a brief formula and cite the source values. Every factual sentence must "
        "include at least one source citation like [S1]. Never cite unsupported sources."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved context (Treasury Bulletin):\n{context_block}\n\n"
        "Return only the grounded answer with citations, or NOT_FOUND."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def format_context_block(snippets: Sequence[SourceSnippet]) -> str:
    """Format source snippets for the model prompt."""
    if not snippets:
        return "[No retrieved Treasury Bulletin context]"
    return "\n\n".join(
        f"[{snippet.label}] {snippet.citation}\n{snippet.text}" for snippet in snippets
    )


def has_source_citation(answer: str) -> bool:
    """Return True when an answer cites at least one retrieved source label."""
    return SOURCE_CITATION_PATTERN.search(answer) is not None


def load_env_file(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs without overwriting existing environment values."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
