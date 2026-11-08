"""Known association effects, patient independence, confounding and missingness."""

import json
import unittest

import numpy as np
from scipy.stats.contingency import association
from sklearn.metrics import adjusted_mutual_info_score

from bp26.attention.associations import (
    DEFAULTS,
    bootstrap_interval,
    cluster_associations,
    cramers_v,
    eta_squared,
    patient_rows,
    property_association,
    site_adjusted,
)


class AssociationTests(unittest.TestCase):
    def test_categorical_effect_matches_reference_and_ignores_label_numbers(self):
        clusters = np.repeat([7, 20], [30, 40])
        values = np.array(["a"] * 20 + ["b"] * 10 + ["a"] * 5 + ["b"] * 35)
        expected = association(np.array([[20, 10], [5, 35]]))
        self.assertAlmostEqual(cramers_v(clusters, values), expected)
        self.assertAlmostEqual(cramers_v(-clusters, values), expected)
        self.assertEqual(cramers_v([0, 0, 1, 1], ["a", "b", "a", "b"]), 0)
        self.assertEqual(cramers_v([0, 0, 1, 1], ["a", "a", "b", "b"]), 1)

    def test_numeric_known_effect_and_degenerate_samples(self):
        self.assertAlmostEqual(eta_squared(np.array([0, 0, 1, 1]), np.array([0, 2, 4, 6])), 0.8)
        for clusters, values in [([], []), ([0], [1]), ([0, 0], [1, 2]), ([0, 1], [2, 2])]:
            self.assertIsNone(eta_squared(np.array(clusters), np.array(values)))
        self.assertIsNone(cramers_v([], []))
        self.assertIsNone(cramers_v([0, 0], ["a", "b"]))
        self.assertIsNone(cramers_v([0, 1], ["a", "a"]))

    def test_each_patient_once_with_dominant_cluster_and_explicit_ties(self):
        labels = [2, 2, 3, -1, 4, 3]
        nodes = [dict(site="site")] * len(labels)
        metadata = [
            dict(patient_id="a", age_months=0, sex="Female"),
            dict(patient_id="a", age_months=0, sex="Female"),
            dict(patient_id="a", age_months=0, sex="Female"),
            dict(patient_id="b", age_months=4, sex="Male"),
            dict(patient_id="b", age_months=4, sex="Female"),
            dict(patient_id=None, age_months=None),
        ]
        rows, coverage = patient_rows(labels, nodes, metadata)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["cluster"], 2)
        self.assertEqual(rows[0]["age_months"], 0)
        self.assertEqual(rows[1]["cluster"], 4)  # assigned cluster wins a tie with noise
        self.assertIsNone(rows[1]["sex"])
        self.assertEqual(coverage["unmapped_slides"], 1)
        self.assertEqual(coverage["multiple_cluster_patients"], 2)
        self.assertEqual(coverage["tied_patients"], 1)

    def test_explicit_unknown_categories_are_missing(self):
        rows, _ = patient_rows(
            [0, 1],
            [dict(site="a"), dict(site="a")],
            [dict(patient_id="a", sex=" Unknown "), dict(patient_id="b", sex="Female")],
        )
        self.assertIsNone(rows[0]["sex"])
        result = property_association(rows, "sex", DEFAULTS, 42)
        self.assertEqual(result["categories"], ["Female"])
        self.assertEqual(result["known"], 1)
        self.assertEqual(result["missing"], 1)

    def test_bootstrap_is_reproducible_and_returns_interval(self):
        clusters = np.repeat([0, 1], 50)
        values = np.array(["a"] * 40 + ["b"] * 10 + ["a"] * 20 + ["b"] * 30)
        first = bootstrap_interval(clusters, values, cramers_v, 100, 42)
        self.assertEqual(first, bootstrap_interval(clusters, values, cramers_v, 100, 42))
        self.assertLess(first[0], cramers_v(clusters, values))
        self.assertGreater(first[1], cramers_v(clusters, values))

    def test_site_confounding_and_within_site_numeric_effect(self):
        sites = np.array(["a"] * 4 + ["b"] * 4)
        clusters = np.array([0, 1, 0, 1] * 2)
        ages = np.array([0, 2, 0, 2, 10, 12, 10, 12])
        self.assertAlmostEqual(site_adjusted(clusters, ages, sites, True), 1)
        self.assertLess(eta_squared(clusters, ages), 0.1)
        confounded = np.array([0] * 4 + [1] * 4)
        self.assertIsNone(site_adjusted(confounded, ages, sites, True))
        self.assertAlmostEqual(site_adjusted(confounded, sites, sites, False), 0)
        self.assertIsNone(site_adjusted(np.array([]), np.array([]), np.array([]), True))

    def test_site_adjustment_handles_many_clusters_without_dense_design(self):
        clusters = np.arange(2000)
        sites = np.array(["a", "b"] * 1000)
        values = np.arange(2000, dtype=float)
        self.assertAlmostEqual(site_adjusted(clusters, values, sites, True), 1)

    def test_distribution_counts_missingness_and_ami(self):
        rows = [
            dict(cluster=i // 10, age_months=i, sex="Female" if i < 10 else "Male", site="a")
            for i in range(20)
        ]
        result = property_association(rows, "sex", DEFAULTS, 42)
        self.assertEqual(result["value"], 1)
        self.assertAlmostEqual(
            result["ami"], adjusted_mutual_info_score([0] * 10 + [1] * 10, [0] * 10 + [1] * 10)
        )
        self.assertEqual(result["known"], 20)
        self.assertEqual(sum(sum(row["counts"]) for row in result["distributions"]), 20)
        rows[0]["sex"] = None
        result = property_association(rows, "sex", DEFAULTS, 42)
        self.assertEqual(result["missing"], 1)
        self.assertEqual(result["distributions"][0]["missing"], 1)
        self.assertIsNotNone(result["missingness_v"])
        numeric = property_association(rows, "age_months", DEFAULTS, 42)
        self.assertEqual(sum(sum(row["counts"]) for row in numeric["distributions"]), 20)
        self.assertIsNone(numeric["ami"])

    def test_site_noise_filters_and_no_metadata_remain_valid_json(self):
        labels = [0, 1, -1, 1]
        nodes = [dict(site=site) for site in ["a", "b", "a", "b"]]
        metadata = [
            dict(patient_id=str(i), age_months=age, sex=None, age_group=None)
            for i, age in enumerate([0, 2, None, 8])
        ]
        result = cluster_associations(labels, nodes, metadata, DEFAULTS, 42)
        self.assertEqual(result["cohort"]["included"], 3)
        self.assertEqual(result["cohort"]["noise_patients"], 1)
        self.assertIsNone(result["properties"]["sex"]["value"])
        self.assertEqual(result["properties"]["sex"]["missing"], 3)
        filtered = cluster_associations(labels, nodes, metadata, DEFAULTS, 42, "a", True)
        self.assertEqual(filtered["cohort"]["included"], 2)
        self.assertEqual(
            [row["cluster"] for row in filtered["properties"]["site"]["distributions"]], [-1, 0]
        )
        empty = cluster_associations(labels, nodes, [dict(patient_id=None)] * 4, DEFAULTS, 42)
        self.assertEqual(empty["cohort"]["included"], 0)
        json.dumps(empty, allow_nan=False)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
