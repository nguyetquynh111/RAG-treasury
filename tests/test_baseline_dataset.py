import tempfile
import unittest
from pathlib import Path

import pandas as pd

from baseline.dataset import extract_year_month, load_filtered_officeqa, load_treasury_documents


class BaselineDatasetTests(unittest.TestCase):
    def test_extract_year_month_supports_numeric_and_month_name_patterns(self):
        self.assertEqual(extract_year_month("data/2024_09_treasurybulletin.txt"), (2024, 9))
        self.assertEqual(extract_year_month("treasury_bulletin_1941_01.txt"), (1941, 1))
        self.assertEqual(extract_year_month("data/2025_12_treasurybulletin.txt"), (2025, 12))
        self.assertEqual(extract_year_month("Treasury Bulletin March 2023"), (2023, 3))
        self.assertEqual(extract_year_month("Treasury Bulletin March 1982"), (1982, 3))
        self.assertIsNone(extract_year_month("treasury_bulletin.txt"))

    def test_load_treasury_documents_loads_txt_only_and_filters_selected_years(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "2024_03_treasurybulletin.txt").write_text("Treasury March text", encoding="utf-8")
            (root / "2025_06_treasurybulletin.txt").write_text("Treasury June text", encoding="utf-8")
            (root / "2021_03_treasurybulletin.txt").write_text("Old text", encoding="utf-8")
            (root / "2024_03_notes.md").write_text("Markdown is ignored", encoding="utf-8")

            documents = load_treasury_documents(root, [2024, 2025])

        self.assertEqual(len(documents), 2)
        self.assertEqual([(doc.year, doc.month) for doc in documents], [(2024, 3), (2025, 6)])
        self.assertTrue(all(doc.source_path.endswith(".txt") for doc in documents))

    def test_load_treasury_documents_fails_when_selected_file_lacks_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "2024_03_treasurybulletin.txt").write_text("Treasury text", encoding="utf-8")
            (root / "missing_date.txt").write_text("No metadata", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Could not extract year/month"):
                load_treasury_documents(root, [2024])

    def test_load_filtered_officeqa_filters_rows_and_adds_baseline_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "officeqa_full.csv"
            pd.DataFrame(
                [
                    {
                        "question": "What was reported in March 2024?",
                        "answer": "Answer 1",
                        "source_path": "data/2024_03_treasurybulletin.txt",
                    },
                    {
                        "question": "What was reported in March 2021?",
                        "answer": "Answer 2",
                        "source_files": "treasury_bulletin_2021_03.txt",
                    },
                ]
            ).to_csv(csv_path, index=False)

            rows = load_filtered_officeqa(csv_path, [2024])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.loc[0, "baseline_question"], "What was reported in March 2024?")
        self.assertEqual(rows.loc[0, "baseline_answer"], "Answer 1")
        self.assertEqual(rows.loc[0, "baseline_year"], 2024)
        self.assertEqual(rows.loc[0, "baseline_month"], 3)


if __name__ == "__main__":
    unittest.main()
