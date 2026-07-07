import unittest

from engineered.chunking import (
    chunk_documents,
    split_heading_sections,
    split_table_aware_windows,
    split_tokens,
)
from engineered.dataset import TreasuryDocument


class EngineeredChunkingTests(unittest.TestCase):
    def test_split_heading_sections_preserves_headings(self):
        text = "Preface text\n\n# Fiscal Operations\nBody one\n\n## Debt\nBody two"

        sections = split_heading_sections(text)

        self.assertEqual(sections[0], ("Document Preface", "Preface text"))
        self.assertEqual(sections[1], ("Fiscal Operations", "Fiscal Operations\nBody one"))
        self.assertEqual(sections[2], ("Debt", "Debt\nBody two"))

    def test_split_tokens_uses_overlap_and_skips_empty(self):
        chunks = split_tokens("one two three four five six", chunk_size=4, chunk_overlap=2)

        self.assertEqual(chunks, ["one two three four", "three four five six"])
        self.assertEqual(split_tokens("   ", chunk_size=4, chunk_overlap=2), [])

    def test_chunk_documents_adds_required_metadata(self):
        document = TreasuryDocument(
            text="# March Heading\none two three four five six seven eight nine",
            year=2024,
            month=3,
            source_path="data/2024_03_treasurybulletin.txt",
        )

        chunks = chunk_documents([document], chunk_size=5, chunk_overlap=1)

        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertTrue(chunk.text.strip())
            self.assertEqual(chunk.metadata["year"], 2024)
            self.assertEqual(chunk.metadata["month"], 3)
            self.assertEqual(chunk.metadata["source_path"], "data/2024_03_treasurybulletin.txt")
            self.assertEqual(chunk.metadata["heading"], "March Heading")
            self.assertIn(chunk.metadata["content_type"], {"text", "mixed"})
            self.assertRegex(chunk.metadata["chunk_id"], r"^2024_03_0000_000\d$")

    def test_split_table_aware_windows_preserves_table_rows(self):
        text = (
            "FD-1-Summary of Federal Debt\n"
            "Debt held by public        1,000       2,000\n"
            "Intragovernmental debt     3,000       4,000\n\n"
            "Short narrative sentence about the table."
        )

        windows = split_table_aware_windows(text, chunk_size=40, chunk_overlap=5)

        self.assertTrue(any(content_type in {"table", "mixed"} for _, content_type in windows))
        self.assertIn("Debt held by public", windows[0][0])
        self.assertIn("Intragovernmental debt", windows[0][0])


if __name__ == "__main__":
    unittest.main()
