"""Check the EDA report's chart helpers on small fixtures."""

import unittest

import pandas as pd

from bp26.eda_report import bars, facet_series, stacked


class EdaReportTests(unittest.TestCase):
    def test_facet_series_folds_tail_and_renames(self):
        rows = [["HE", 50], ["(NOT IN FILENAME)", 30], ["IHC", 10], ["GRAM", 6], ["PAS", 4]]
        series = facet_series(rows, limit=3)
        self.assertEqual(series.to_dict(), {"HE": 50, "No stain token": 30, "Other": 20})

    def test_bars_scale_to_largest_and_show_share(self):
        out = bars(pd.Series({"a": 30, "b": 10}))
        self.assertIn("width:100.0%", out)
        self.assertIn("width:33.3%", out)
        self.assertIn("<small>75%</small>", out)

    def test_stacked_puts_remaining_cases_in_other(self):
        causes = dict(
            diseases=[["X", 7], ["Y", 2]],
            sites={"KE": dict(n=6, counts={"X": 4, "Y": 1}), "ML": dict(n=3, counts={"X": 3})},
        )
        out = stacked(causes, {"KE": "Kenya"}, k=1)
        self.assertLess(out.index("Kenya"), out.index("ML"))  # sorted by cases; unknown code kept
        self.assertIn("flex:4;background:var(--s1)", out)
        self.assertIn("flex:2;background:var(--other)", out)


if __name__ == "__main__":
    unittest.main()
