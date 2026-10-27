"""Embedding-space clustering, safe clinical joins and patient-weighted summaries."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from bp26.attention.clustering import DEFAULTS, ClusterStore, parameters
from bp26.attention.metadata import age_months, cluster_summaries, load_metadata


class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(42)
        vectors = np.vstack(
            [rng.normal([1, 0, 0], 0.025, (30, 3)), rng.normal([0, 1, 0], 0.025, (30, 3))]
        ).astype("float32")
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        distances = 1 - vectors @ vectors.T
        np.fill_diagonal(distances, np.inf)
        nodes = [
            dict(id=str(i), case_id=str(i // 2), site="site", organ="liver", slide_id=str(i))
            for i in range(60)
        ]
        self.graph = SimpleNamespace(
            config=dict(
                cpu_threads=1, data_root=self.root, slide_registry=self.root / "missing.parquet"
            ),
            settings=dict(seed=42, cache_root=str(self.root)),
            dataset=lambda *args: dict(vectors=vectors, nodes=nodes, digest="test"),
            rankings=lambda *args: np.argsort(distances, kind="stable")[:, :-1],
        )
        self.store = ClusterStore(self.graph)

    def test_all_three_methods_separate_known_groups_and_preserve_node_order(self):
        for method, options in [
            ("kmeans", dict(n_clusters=2)),
            ("hdbscan", dict(min_cluster_size=10, min_samples=5)),
            ("leiden", dict(leiden_k=15, resolution=0.5)),
        ]:
            with self.subTest(method=method):
                result = self.store.run("liver", 0, method, options)
                self.assertEqual(len(result["labels"]), 60)
                self.assertEqual(result["n_clusters"], 2)
                self.assertEqual(len(set(result["labels"][:30])), 1)
                self.assertEqual(len(set(result["labels"][30:])), 1)
                self.assertNotEqual(result["labels"][0], result["labels"][30])
                self.assertEqual(sum(row["slides"] for row in result["summaries"]), 60)

    def test_cache_and_seed(self):
        first = self.store.run("liver", 0, "kmeans", dict(n_clusters=2))
        with patch(
            "bp26.attention.clustering.KMeans", side_effect=AssertionError("cache not used")
        ):
            second = self.store.run("liver", 0, "kmeans", dict(n_clusters=2))
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], 42)

    def test_invalid_and_method_specific_parameters(self):
        for method, values in [
            ("invalid", {}),
            ("kmeans", dict(n_clusters="2.5")),
            ("kmeans", dict(n_clusters=61)),
            ("leiden", dict(resolution="nan")),
            ("leiden", dict(leiden_k=0)),
            ("hdbscan", dict(min_cluster_size=1)),
        ]:
            with self.subTest(method=method, values=values), self.assertRaises(ValueError):
                parameters(method, values, 60, DEFAULTS)
        self.assertEqual(
            parameters("kmeans", dict(n_clusters="2", leiden_k="bad"), 60, DEFAULTS),
            dict(n_clusters=2),
        )
        with self.assertRaises(ValueError):
            parameters("leiden", dict(leiden_k=20), 60, DEFAULTS | dict(max_leiden_edges=600))

    def test_noise_is_explicit(self):
        with patch("bp26.attention.clustering.HDBSCAN") as estimator:
            estimator.return_value.fit_predict.return_value = np.array([-1] * 60)
            result = self.store.run("liver", 0, "hdbscan", dict(min_cluster_size=10, min_samples=5))
        self.assertEqual(result["n_clusters"], 0)
        self.assertEqual(result["noise_count"], 60)
        self.assertEqual(result["summaries"][0]["cluster"], -1)


class MetadataTests(unittest.TestCase):
    def test_age_components_and_missing_values(self):
        self.assertAlmostEqual(
            age_months(dict(age_years="2", age_months="3", age_days="15", age_hours="12")),
            27 + 15.5 * 12 / 365.25,
        )
        self.assertEqual(age_months(dict(age_hours="0")), 0)
        self.assertIsNone(age_months({}))
        self.assertIsNone(age_months(dict(age_days="-1")))
        self.assertIsNone(age_months(dict(age_months="unknown")))

    def test_blank_and_ambiguous_ids_do_not_join(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                [
                    dict(study_id="case", champs_deid="patient"),
                    dict(study_id="conflict", champs_deid="one"),
                    dict(study_id="conflict", champs_deid="two"),
                    dict(study_id="", champs_deid="patient"),
                ]
            ).to_csv(root / "mapping.csv", index=False)
            pd.DataFrame(
                [dict(champs_deid="patient", age_years="1", age_months="2", sex="Female")]
            ).to_csv(root / "demographics.csv", index=False)
            config = dict(
                data_root=root,
                slide_registry=root / "missing.parquet",
                graph_metadata=dict(mapping="mapping.csv", demographics="demographics.csv"),
            )
            rows = load_metadata(
                config,
                [
                    dict(id=str(i), case_id=case)
                    for i, case in enumerate(["case", "conflict", "", "missing"])
                ],
            )["records"]
            self.assertEqual(rows[0]["age_months"], 14)
            self.assertEqual(rows[0]["sex"], "Female")
            self.assertTrue(
                all(row["patient_id"] is None and row["age_months"] is None for row in rows[1:])
            )

    def test_patient_age_summary_counts_each_patient_once(self):
        nodes = [dict(site="site")] * 4
        metadata = [
            dict(patient_id="a", age_months=1),
            dict(patient_id="a", age_months=1),
            dict(patient_id="b", age_months=9),
            dict(patient_id=None, age_months=None),
        ]
        result = cluster_summaries([0, 0, 0, 0], nodes, metadata)[0]
        self.assertEqual(result["slides"], 4)
        self.assertEqual(result["patients"], 2)
        self.assertEqual(result["median_age_months"], 5)
        self.assertEqual(sum(result["age_bins"]), 2)
        self.assertEqual(result["unmapped_slides"], 1)


if __name__ == "__main__":
    unittest.main()
