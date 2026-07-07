import tempfile
import unittest
from pathlib import Path

import pandas as pd

from engineered.dataset import extract_year_month, load_filtered_officeqa, load_treasury_documents


class EngineeredDatasetTests(unittest.TestCase):
    def test_extract_year_month_supports_filename_and_month_name(self):
        self.assertEqual(extract_year_month("data/2024_09_treasurybulletin.txt"), (2024, 9))
        self.assertEqual(extract_year_month("March 2025 Treasury Bulletin"), (2025, 3))
        self.assertEqual(extract_year_month("treasury_bulletin_1941_01.txt"), (1941, 1))

    def test_load_treasury_documents_filters_years_and_fails_on_bad_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "2024_03_treasurybulletin.txt").write_text("Treasury text", encoding="utf-8")
            (root / "2024_04_notes.md").write_text("Ignored note", encoding="utf-8")
            (root / "2021_03_treasurybulletin.txt").write_text("Old text", encoding="utf-8")
            (root / "bad_file.txt").write_text("Bad metadata", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_treasury_documents(root, [2024])

            (root / "bad_file.txt").unlink()
            documents = load_treasury_documents(root, [2024])

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].year, 2024)
        self.assertEqual(documents[0].month, 3)

    def test_load_filtered_officeqa_uses_selected_years(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "officeqa_full.csv"
            pd.DataFrame(
                [
                    {
                        "uid": "q1",
                        "question": "What was reported in March 2024?",
                        "answer": "Answer 1",
                        "source_files": "treasury_bulletin_2024_03.txt",
                    },
                    {
                        "uid": "q2",
                        "question": "What was reported in March 2021?",
                        "answer": "Answer 2",
                        "source_files": "treasury_bulletin_2021_03.txt",
                    },
                ]
            ).to_csv(csv_path, index=False)

            rows = load_filtered_officeqa(csv_path, [2024])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].question_id, "q1")
        self.assertEqual(rows[0].row_year, 2024)
        self.assertEqual(rows[0].row_month, 3)


if __name__ == "__main__":
    unittest.main()
