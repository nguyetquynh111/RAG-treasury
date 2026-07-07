"""Shared section-aware chunking for Treasury text documents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from common.text import split_tokens


MARKDOWN_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*$")
TABLE_HEADING_PATTERN = re.compile(r"^(table|chart|schedule)\s+[A-Z0-9IVXLC\-.]+\b", re.IGNORECASE)
MIN_TABLE_COLUMNS = 3


@dataclass(frozen=True)
class TextChunk:
    """One chunk saved in the shared index."""

    text: str
    metadata: dict[str, int | str]


@dataclass(frozen=True)
class TextBlock:
    """A paragraph, heading, or table-like block with its nearest heading."""

    text: str
    heading: str
    content_type: str


@dataclass(frozen=True)
class ChunkContext:
    """Metadata context passed from the shared chunker into pipeline-specific metadata."""

    document: Any
    chunk_number: int
    text: str
    heading: str
    content_type: str


MetadataBuilder = Callable[[ChunkContext], dict[str, Any]]


def chunk_documents(
    documents: list[Any],
    chunk_size: int,
    chunk_overlap: int,
    *,
    build_metadata: MetadataBuilder,
) -> list[TextChunk]:
    """Split documents into reusable chunks and attach caller-provided metadata.

    The chunker is shared by baseline and engineered RAG. It keeps nearby headings
    with paragraphs, avoids breaking small tables across chunks when possible, and
    falls back to overlapping token windows for long blocks.
    """
    chunks: list[TextChunk] = []
    for document in documents:
        text_chunks = split_treasury_text(document.text, chunk_size, chunk_overlap)
        if not text_chunks:
            raise ValueError(f"No chunks produced for Treasury document: {document.source_path}")

        for chunk_number, context in enumerate(text_chunks):
            chunk_context = ChunkContext(
                document=document,
                chunk_number=chunk_number,
                text=context.text,
                heading=context.heading,
                content_type=context.content_type,
            )
            chunks.append(
                TextChunk(
                    text=context.text,
                    metadata=build_metadata(chunk_context),
                )
            )

    if not chunks:
        raise ValueError("No chunks were created from the selected Treasury documents.")
    return chunks


def split_treasury_text(text: str, chunk_size: int, chunk_overlap: int) -> list[TextBlock]:
    """Split Treasury text using section and table boundaries before token windows."""
    validate_chunk_settings(chunk_size, chunk_overlap)
    blocks = parse_text_blocks(text)
    if not blocks:
        return []

    chunks: list[TextBlock] = []
    current_tokens: list[str] = []
    current_heading = blocks[0].heading
    current_content_type = blocks[0].content_type

    for block in blocks:
        block_tokens = block.text.split()
        if not block_tokens:
            continue

        if len(block_tokens) > chunk_size:
            if current_tokens:
                chunks.append(
                    TextBlock(
                        text=" ".join(current_tokens),
                        heading=current_heading,
                        content_type=current_content_type,
                    )
                )
                current_tokens = []
            chunks.extend(split_long_block(block, chunk_size, chunk_overlap))
            current_heading = block.heading
            current_content_type = block.content_type
            continue

        if current_tokens and len(current_tokens) + len(block_tokens) > chunk_size:
            chunks.append(
                TextBlock(
                    text=" ".join(current_tokens),
                    heading=current_heading,
                    content_type=current_content_type,
                )
            )
            overlap_tokens = current_tokens[-chunk_overlap:] if chunk_overlap else []
            current_tokens = overlap_tokens + block_tokens
            current_heading = block.heading
            current_content_type = merge_content_type(current_content_type, block.content_type)
            continue

        if not current_tokens:
            current_heading = block.heading
            current_content_type = block.content_type
        else:
            current_content_type = merge_content_type(current_content_type, block.content_type)
        current_tokens.extend(block_tokens)

    if current_tokens:
        chunks.append(
            TextBlock(
                text=" ".join(current_tokens),
                heading=current_heading,
                content_type=current_content_type,
            )
        )
    return chunks


def parse_text_blocks(text: str) -> list[TextBlock]:
    """Parse text into heading-aware paragraph/table blocks."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    heading = "Document"
    pending_lines: list[str] = []
    blocks: list[TextBlock] = []

    def flush() -> None:
        if not pending_lines:
            return
        block_text = "\n".join(pending_lines).strip()
        pending_lines.clear()
        if block_text:
            blocks.append(TextBlock(block_text, heading, detect_content_type(block_text)))

    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            flush()
            continue

        detected_heading = clean_heading(line)
        if detected_heading is not None:
            flush()
            heading = detected_heading
            pending_lines.append(detected_heading)
            continue

        pending_lines.append(line)

    flush()
    return blocks


def split_long_block(block: TextBlock, chunk_size: int, chunk_overlap: int) -> list[TextBlock]:
    """Split one oversized block with the same stable heading/content metadata."""
    return [
        TextBlock(text=chunk_text, heading=block.heading, content_type=block.content_type)
        for chunk_text in split_tokens(block.text, chunk_size, chunk_overlap)
    ]


def clean_heading(line: str) -> str | None:
    """Return a normalized heading when a line looks like a section/table heading."""
    match = MARKDOWN_HEADING_PATTERN.match(line)
    if match:
        return normalize_inline_text(match.group(1))

    compact = normalize_inline_text(line.lstrip("#"))
    if not compact or len(compact) > 180:
        return None
    if TABLE_HEADING_PATTERN.match(compact):
        return compact
    if looks_like_title_heading(compact):
        return compact
    return None


def looks_like_title_heading(line: str) -> bool:
    """Detect common Treasury extracted-text headings without treating rows as headings."""
    if line.endswith(('.', ',', ';', ':')):
        return False
    if count_numeric_tokens(line) > 1:
        return False
    letters = [char for char in line if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    words = line.split()
    return uppercase_ratio >= 0.75 and 1 <= len(words) <= 14


def detect_content_type(text: str) -> str:
    """Label table-like blocks so engineered metadata can cite them."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return "text"
    table_like_lines = sum(1 for line in lines if is_table_like_line(line))
    return "table" if table_like_lines >= 2 else "text"


def is_table_like_line(line: str) -> bool:
    """Heuristic for extracted Treasury rows with multiple numeric columns."""
    has_column_spacing = bool(re.search(r"\S\s{2,}\S", line))
    numeric_tokens = count_numeric_tokens(line)
    return has_column_spacing and numeric_tokens >= MIN_TABLE_COLUMNS


def count_numeric_tokens(text: str) -> int:
    """Count number-like tokens in a line."""
    return len(re.findall(r"[$-]?\d[\d,]*(?:\.\d+)?%?", text))


def merge_content_type(left: str, right: str) -> str:
    """Prefer table metadata when any block in a chunk is table-like."""
    return "table" if "table" in {left, right} else "text"


def base_chunk_metadata(context: ChunkContext | Any, chunk_number: int | None = None) -> dict[str, int | str]:
    """Return the required retrieval metadata for one chunk.

    Accepts the new ChunkContext object and the older (document, chunk_number)
    call shape for compatibility with existing tests or notebooks.
    """
    chunk_context = normalize_chunk_context(context, chunk_number)
    document = chunk_context.document
    chunk_id = f"{document.year}_{document.month:02d}_{chunk_context.chunk_number:05d}"
    return {
        "year": document.year,
        "month": document.month,
        "source_path": document.source_path,
        "chunk_id": chunk_id,
    }


def full_chunk_metadata(context: ChunkContext | Any, chunk_number: int | None = None) -> dict[str, int | str]:
    """Return the shared index metadata superset used by both pipelines.

    Baseline retrieval ignores the descriptive fields. Engineered retrieval uses
    them for metadata filtering and audit output. Keeping this superset in the
    single shared index prevents baseline/engineered index drift.
    """
    chunk_context = normalize_chunk_context(context, chunk_number)
    metadata = base_chunk_metadata(chunk_context)
    metadata.update(
        {
            "heading": chunk_context.heading,
            "content_type": chunk_context.content_type,
        }
    )
    return metadata


def normalize_chunk_context(context: ChunkContext | Any, chunk_number: int | None = None) -> ChunkContext:
    """Coerce old and new metadata-builder call shapes into ChunkContext."""
    if isinstance(context, ChunkContext):
        return context
    if chunk_number is None:
        raise TypeError("chunk_number is required when passing a document directly.")
    blocks = parse_text_blocks(context.text)
    heading = blocks[0].heading if blocks else "Document"
    content_type = blocks[0].content_type if blocks else "text"
    return ChunkContext(
        document=context,
        chunk_number=chunk_number,
        text=context.text,
        heading=heading,
        content_type=content_type,
    )


def save_chunks(chunks: list[TextChunk], output_path: str | Path) -> None:
    """Save chunks as JSONL records."""
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {"text": chunk.text, "metadata": chunk.metadata}
            handle.write(json.dumps(record) + "\n")


def load_chunks(
    chunks_path: str | Path,
    *,
    required_metadata_keys: Iterable[str] = (),
    not_found_hint: str = "Run indexing first.",
) -> list[dict[str, Any]]:
    """Load chunk JSONL records and validate required metadata."""
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}. {not_found_hint}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
            validate_chunk_record(
                record,
                path=path,
                line_number=line_number,
                required_metadata_keys=required_metadata_keys,
            )
            records.append(record)

    if not records:
        raise ValueError(f"No chunk records found in {path}")
    return records


def validate_chunk_record(
    record: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    required_metadata_keys: Iterable[str],
) -> None:
    """Validate one chunk record from chunks.jsonl."""
    if "text" not in record or "metadata" not in record:
        raise ValueError(f"Chunk record missing text/metadata in {path} line {line_number}")
    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"Chunk metadata must be a mapping in {path} line {line_number}")
    for key in required_metadata_keys:
        if key not in metadata:
            raise ValueError(f"Chunk metadata missing {key!r} in {path} line {line_number}")


def normalize_inline_text(value: str) -> str:
    """Collapse inline whitespace."""
    return " ".join(str(value).split())


def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Validate shared chunking settings."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
