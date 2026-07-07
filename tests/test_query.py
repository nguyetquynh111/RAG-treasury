import unittest

from common.query import QueryFilters, parse_query_filters


class QueryFilterTests(unittest.TestCase):
    def test_parse_month_year(self):
        self.assertEqual(
            parse_query_filters("What happened in March 2024?", [2024]),
            QueryFilters(year=2024, month=3),
        )

    def test_parse_numeric_month_year(self):
        self.assertEqual(
            parse_query_filters("What happened in 2024-09?", [2024]),
            QueryFilters(year=2024, month=9),
        )

    def test_missing_year_returns_empty_filter(self):
        self.assertEqual(parse_query_filters("What happened in March?", [2024]), QueryFilters(None, None))


if __name__ == "__main__":
    unittest.main()
