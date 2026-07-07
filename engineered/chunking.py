"""Section- and table-aware chunking for Treasury text documents."""

from __future__ import annotations

import re

from common.chunks import TextChunk
from common.text import split_tokens
from engineered.dataset import TreasuryDocument


HASH_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
TABLE_TITLE_PATTERN = re.compile(r"^(?:TABLE\s+)?[A-Z]{1,6}(?:-[A-Z0-9]+)+[A-Z]?[.—-].+")
INTRO_ANALYSIS_PATTERN = re.compile(r"^(?:Introduction|Analysis)[—-].+")
SECTION_HEADING_PATTERN = re.compile(r"^[A-Z][A-Z0-9&.,'() /-]{6,}$")
DOT_LEADER_PATTERN = re.compile(r"\.{4,}")
MANY_SPACES_PATTERN = re.compile(r"\S\s{2,}\S")


def chunk_documents(
    documents: list[TreasuryDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Split documents by Treasury sections and table-aware token windows."""
    chunks: list[TextChunk] = []
    for document in documents:
        sections = split_heading_sections(document.text)
        if not sections:
            raise ValueError(f"No sections produced for Treasury document: {document.source_path}")

        section_chunk_count = 0
        for section_index, (heading, section_text) in enumerate(sections):
            windows = split_table_aware_windows(section_text, chunk_size, chunk_overlap)
            for window_index, (window_text, content_type) in enumerate(windows):
                if not window_text.strip():
                    continue
                chunk_id = f"{document.year}_{document.month:02d}_{section_index:04d}_{window_index:04d}"
                chunks.append(
                    TextChunk(
                        text=window_text,
                        metadata={
                            "year": document.year,
                            "month": document.month,
                            "source_path": document.source_path,
                            "heading": heading,
                            "content_type": content_type,
                            "chunk_id": chunk_id,
                        },
                    )
                )
                section_chunk_count += 1

        if section_chunk_count == 0:
            raise ValueError(f"No chunks produced for Treasury document: {document.source_path}")

    if not chunks:
        raise ValueError("No chunks were created from the selected Treasury documents.")
    return chunks


def split_heading_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading, text) sections using Treasury or hash-style headings."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(HASH_HEADING_PATTERN.finditer(normalized))
    if not matches:
        return split_treasury_sections(normalized)

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preface = normalized[: matches[0].start()].strip()
        if preface:
            sections.append(("Document Preface", preface))

    for index, match in enumerate(matches):
        heading = " ".join(match.group(2).split())
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        body = normalized[start:end].strip()
        section_text = f"{heading}\n{body}".strip()
        if section_text:
            sections.append((heading, section_text))

    return sections


def split_treasury_sections(text: str) -> list[tuple[str, str]]:
    """Split Treasury Bulletin text at high-signal section and table headings."""
    lines = text.splitlines()
    heading_indices = [
        index
        for index, line in enumerate(lines)
        if is_treasury_heading(line) and not is_table_of_contents_line(line)
    ]

    if not heading_indices:
        fallback_heading = _first_nonempty_line(text) or "Document"
        return [(fallback_heading, text.strip())] if text.strip() else []

    sections: list[tuple[str, str]] = []
    if heading_indices[0] > 0:
        preface = "\n".join(lines[: heading_indices[0]]).strip()
        if preface:
            sections.append(("Document Preface", preface))

    for position, start_index in enumerate(heading_indices):
        end_index = heading_indices[position + 1] if position + 1 < len(heading_indices) else len(lines)
        heading = normalize_heading(lines[start_index])
        body = "\n".join(lines[start_index:end_index]).strip()
        if body:
            sections.append((heading, body))
    return sections


def is_treasury_heading(line: str) -> bool:
    """Return True for Treasury section/table headings worth preserving."""
    clean = normalize_heading(line)
    if not clean or len(clean) > 180:
        return False
    if TABLE_TITLE_PATTERN.match(clean):
        return True
    if INTRO_ANALYSIS_PATTERN.match(clean):
        return True
    if SECTION_HEADING_PATTERN.match(clean) and not looks_like_table_row(clean):
        return True
    return False


def is_table_of_contents_line(line: str) -> bool:
    """Filter out TOC rows that look like headings but only point to pages."""
    clean = normalize_heading(line)
    return bool(DOT_LEADER_PATTERN.search(clean) and re.search(r"\s\d+\s*$", clean))


def split_table_aware_windows(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, str]]:
    """Split text into windows while keeping table-like rows together when possible."""
    blocks = split_content_blocks(text)
    if not blocks:
        return []

    windows: list[tuple[str, str]] = []
    current_blocks: list[tuple[str, str]] = []
    current_tokens = 0

    for block_text, block_type in blocks:
        block_tokens = len(block_text.split())
        if block_tokens > chunk_size:
            if current_blocks:
                windows.append(render_window(current_blocks))
                current_blocks = []
                current_tokens = 0
            windows.extend((chunk, block_type) for chunk in split_tokens(block_text, chunk_size, chunk_overlap))
            continue

        if current_blocks and current_tokens + block_tokens > chunk_size:
            windows.append(render_window(current_blocks))
            current_blocks = overlap_tail(current_blocks, chunk_overlap)
            current_tokens = sum(len(text.split()) for text, _ in current_blocks)

        current_blocks.append((block_text, block_type))
        current_tokens += block_tokens

    if current_blocks:
        windows.append(render_window(current_blocks))

    return [(text, content_type) for text, content_type in windows if text.strip()]


def split_content_blocks(text: str) -> list[tuple[str, str]]:
    """Group prose paragraphs and table-like row runs into typed blocks."""
    blocks: list[tuple[str, str]] = []
    paragraph: list[str] = []
    table: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append((" ".join(line.strip() for line in paragraph).strip(), "text"))
            paragraph = []

    def flush_table() -> None:
        nonlocal table
        if table:
            blocks.append(("\n".join(table).strip(), "table"))
            table = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_table()
            continue
        if looks_like_table_row(line):
            flush_paragraph()
            table.append(line)
        else:
            flush_table()
            paragraph.append(line)

    flush_paragraph()
    flush_table()
    return [(block, block_type) for block, block_type in blocks if block]


def looks_like_table_row(line: str) -> bool:
    """Heuristic for rows extracted from Treasury PDF tables."""
    clean = line.strip()
    if not clean:
        return False
    numeric_terms = len(re.findall(r"(?<![A-Za-z])[-(]?\$?\d[\d,]*(?:\.\d+)?%?\)?", clean))
    has_table_spacing = bool(DOT_LEADER_PATTERN.search(clean) or MANY_SPACES_PATTERN.search(clean))
    return numeric_terms >= 2 or (numeric_terms >= 1 and has_table_spacing)


def render_window(blocks: list[tuple[str, str]]) -> tuple[str, str]:
    """Render grouped blocks and return the dominant content type."""
    text = "\n\n".join(block for block, _ in blocks).strip()
    types = {block_type for _, block_type in blocks}
    content_type = types.pop() if len(types) == 1 else "mixed"
    return text, content_type


def overlap_tail(blocks: list[tuple[str, str]], overlap_tokens: int) -> list[tuple[str, str]]:
    """Keep whole trailing blocks up to the configured overlap budget."""
    if overlap_tokens <= 0:
        return []
    selected: list[tuple[str, str]] = []
    total = 0
    for block in reversed(blocks):
        block_tokens = len(block[0].split())
        if selected and total + block_tokens > overlap_tokens:
            break
        selected.append(block)
        total += block_tokens
        if total >= overlap_tokens:
            break
    return list(reversed(selected))


def normalize_heading(line: str) -> str:
    return " ".join(line.replace("\f", " ").split())


def _first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        clean = " ".join(line.split())
        if clean:
            return clean[:120]
    return None
