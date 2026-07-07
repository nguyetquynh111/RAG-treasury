"""Fixed-size chunking for Treasury text documents."""

from __future__ import annotations

from baseline.dataset import TreasuryDocument
from common.chunks import TextChunk
from common.text import split_tokens


def chunk_documents(
    documents: list[TreasuryDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Chunk Treasury documents and attach required metadata."""
    chunks: list[TextChunk] = []
    for document in documents:
        text_chunks = split_tokens(document.text, chunk_size, chunk_overlap)
        if not text_chunks:
            raise ValueError(f"No chunks produced for Treasury document: {document.source_path}")

        for chunk_number, chunk_text in enumerate(text_chunks):
            chunk_id = f"{document.year}_{document.month:02d}_{chunk_number:05d}"
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    metadata={
                        "year": document.year,
                        "month": document.month,
                        "source_path": document.source_path,
                        "chunk_id": chunk_id,
                    },
                )
            )

    if not chunks:
        raise ValueError("No chunks were created from the selected Treasury documents.")
    return chunks
