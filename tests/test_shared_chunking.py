import unittest

from common.chunking import chunk_documents, full_chunk_metadata
from common.dataset import TreasuryDocument
from common.text import split_tokens


class SharedChunkingTests(unittest.TestCase):
    def test_chunk_documents_keeps_metadata_superset(self):
        documents = [
            TreasuryDocument(
                year=2024,
                month=3,
                source_path="2024_03_treasurybulletin.txt",
                text="# March Bulletin\npublic debt securities",
            )
        ]

        chunks = chunk_documents(
            documents,
            chunk_size=8,
            chunk_overlap=0,
            build_metadata=full_chunk_metadata,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["year"], 2024)
        self.assertEqual(chunks[0].metadata["month"], 3)
        self.assertEqual(chunks[0].metadata["heading"], "March Bulletin")
        self.assertEqual(chunks[0].metadata["content_type"], "text")

    def test_split_tokens_uses_overlap(self):
        chunks = split_tokens("a b c d e", chunk_size=3, chunk_overlap=1)
        self.assertEqual(chunks, ["a b c", "c d e"])


if __name__ == "__main__":
    unittest.main()
