import unittest

from baseline.chunking import chunk_documents, split_tokens
from baseline.dataset import TreasuryDocument


class BaselineChunkingTests(unittest.TestCase):
    def test_split_tokens_uses_fixed_size_and_overlap(self):
        chunks = split_tokens("one two three four five six", chunk_size=3, chunk_overlap=1)

        self.assertEqual(chunks, ["one two three", "three four five", "five six"])

    def test_split_tokens_rejects_invalid_window_settings(self):
        with self.assertRaises(ValueError):
            split_tokens("one two", chunk_size=2, chunk_overlap=2)

    def test_chunk_documents_attaches_required_metadata(self):
        document = TreasuryDocument(
            text="alpha beta gamma delta epsilon",
            year=2024,
            month=9,
            source_path="data/2024_09_treasurybulletin.txt",
        )

        chunks = chunk_documents([document], chunk_size=3, chunk_overlap=1)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["year"], 2024)
        self.assertEqual(chunks[0].metadata["month"], 9)
        self.assertEqual(chunks[0].metadata["source_path"], document.source_path)
        self.assertEqual(chunks[0].metadata["chunk_id"], "2024_09_00000")


if __name__ == "__main__":
    unittest.main()
