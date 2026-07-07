import unittest

from engineered.query import QueryFilters, parse_query_filters


class EngineeredQueryTests(unittest.TestCase):
    def test_parse_month_name_and_year(self):
        self.assertEqual(
            parse_query_filters("What happened in March 2024?", [2022, 2023, 2024, 2025]),
            QueryFilters(year=2024, month=3),
        )

    def test_parse_year_only(self):
        self.assertEqual(
            parse_query_filters("Summarize the 2023 bulletin.", [2022, 2023, 2024, 2025]),
            QueryFilters(year=2023, month=None),
        )

    def test_parse_numeric_month(self):
        self.assertEqual(
            parse_query_filters("What was listed for 2025-09?", [2022, 2023, 2024, 2025]),
            QueryFilters(year=2025, month=9),
        )

    def test_no_date_returns_no_filters(self):
        self.assertEqual(
            parse_query_filters("What was the reported total?", [2022, 2023, 2024, 2025]),
            QueryFilters(year=None, month=None),
        )


if __name__ == "__main__":
    unittest.main()
