"""Graph identity joins, exact high-dimensional neighbors and API validation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from bp26.attention.graph import GraphStore
from bp26.attention.server import create_app


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(
            data_root=self.root,
            models_root=self.root,
            slide_registry=self.root / "registry.parquet",
            attention_cache_size=1,
            open_slide_cache_size=1,
            cpu_threads=1,
            graph=dict(
                embeddings_root=str(self.root),
                cache_root=str(self.root / "cache"),
                organs=["liver"],
                seed=42,
                perplexity=2,
                tsne_iterations=250,
                edge_budget=4,
            ),
        )
        self.vectors = np.array(
            [[1, 0, 0], [0.8, 0.2, 0], [0, 1, 0], [-1, 0, 0], [np.nan, 0, 0]], dtype="float32"
        )
        with h5py.File(self.root / "liver.h5", "w") as handle:
            handle.create_dataset("embeddings", data=self.vectors[:, None, :])
            handle.create_dataset(
                "slide_ids", data=["c", "a", "b", "d", "unfinished"], dtype=h5py.string_dtype()
            )
            handle.create_dataset("branch_labels", data=["finding"], dtype=h5py.string_dtype())
            handle.create_dataset("completed", data=[True, True, True, True, False])
            handle.attrs.update(
                organ="liver",
                encoder="virchow2",
                checkpoint_sha256="checkpoint",
                manifest_sha256="manifest",
            )
        self.catalog = pd.DataFrame(
            [
                dict(
                    id=f"key-{sid}",
                    slide_id=sid,
                    organ="liver",
                    case_id=f"case-{sid}",
                    site="site",
                    path="unused",
                )
                for sid in ["a", "b", "c", "d", "unfinished"]
            ]
        ).set_index("id", drop=False)
        self.store = GraphStore(self.config, self.catalog)
        self.addCleanup(self.store.executor.shutdown)

    def test_identity_join_and_exact_neighbors_exclude_self_and_incomplete(self):
        data = self.store.dataset("liver", 0)
        self.assertEqual(
            [node["id"] for node in data["nodes"]], ["key-c", "key-a", "key-b", "key-d"]
        )
        self.assertEqual(data["excluded"], 1)
        result = self.store.neighbors("liver", 0, "key-c", 3)
        self.assertEqual(result["indices"], [1, 2, 3])
        np.testing.assert_allclose(
            result["distances"], [1 - 0.8 / np.hypot(0.8, 0.2), 1, 2], atol=1e-6
        )
        self.assertEqual(self.store.neighbors("liver", 0, "key-c", 1)["indices"], [1])
        for k in (0, 4, -1):
            with self.assertRaises(ValueError):
                self.store.neighbors("liver", 0, "key-c", k)

    def test_projection_preserves_neighbors_and_cache(self):
        before = self.store.neighbors("liver", 0, "key-a", 3)
        for projection in ("pca", "tsne", "umap"):
            result = self.store.projection("liver", 0, projection)
            self.assertEqual(np.asarray(result["coords"]).shape, (4, 2))
            self.assertTrue(np.isfinite(result["coords"]).all())
            self.assertEqual(result["nodes"][0]["slide_id"], "c")
        self.assertEqual(before, self.store.neighbors("liver", 0, "key-a", 3))
        with patch("bp26.attention.graph.PCA", side_effect=AssertionError("cache not used")):
            for projection in ("pca", "tsne", "umap"):
                self.store.projection("liver", 0, projection)

    def test_graph_rank_cache_supports_arbitrary_k_and_discloses_sample(self):
        first = self.store.edges("liver", 0, 1)
        self.assertFalse(first["sampled"])
        self.assertEqual(first["total_edges"], 4)
        for source, target in first["edges"]:
            self.assertNotEqual(source, target)
            self.assertEqual(
                target,
                self.store.neighbors(
                    "liver", 0, self.store.dataset("liver", 0)["nodes"][source]["id"], 1
                )["indices"][0],
            )
        dense = self.store.edges("liver", 0, 3)
        self.assertTrue(dense["sampled"])
        self.assertEqual(dense["total_edges"], 12)
        self.assertEqual(dense["displayed_edges"], 4)
        self.assertEqual(len(list((self.root / "cache").glob("*-neighbors.npy"))), 1)

    def test_missing_slides_recovered_in_catalog_order(self):
        self.config["graph"]["recover_missing"] = True
        self.store.catalog = pd.concat(
            [
                self.catalog,
                pd.DataFrame(
                    [
                        dict(
                            id="key-extra",
                            slide_id="extra",
                            organ="liver",
                            case_id="case-extra",
                            site="site",
                            path="unused",
                        )
                    ]
                ).set_index("id", drop=False),
            ]
        )
        with patch.object(
            self.store, "recover", return_value=np.array([[[0, 0, 1]]], dtype="float32")
        ) as recover:
            data = self.store.dataset("liver", 0)
        recover.assert_called_once()
        self.assertEqual(data["nodes"][-1]["id"], "key-extra")
        self.assertEqual(data["provenance"]["recovered_slides"], ["extra"])
        self.assertEqual(data["excluded"], 1)

    def test_association_api_validation_and_completed_response(self):
        app = create_app(self.config, self.catalog)
        graph = app.extensions["graph_store"]
        self.addCleanup(graph.executor.shutdown)
        client = app.test_client()
        for query in (
            "method=invalid",
            "method=kmeans&n_clusters=2&include_noise=maybe",
            "method=kmeans&n_clusters=99",
        ):
            self.assertEqual(client.get(f"/api/graph/associations?{query}").status_code, 400)
        url = "/api/graph/associations?method=kmeans&n_clusters=2"
        self.assertIn(client.get(url).status_code, (200, 202))
        graph.executor.submit(lambda: None).result(timeout=20)
        response = client.get(url)
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["unit"], "patient")
        self.assertEqual(result["cohort"]["unmapped_slides"], 4)
        self.assertEqual(result["cohort"]["included"], 0)
        self.assertEqual(set(result["properties"]), {"age_months", "sex", "site", "age_group"})
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(client.get(url).get_json(), result)

    def test_api_invalid_k_branch_projection_and_static_view(self):
        app = create_app(self.config, self.catalog)
        self.addCleanup(app.extensions["graph_store"].executor.shutdown)
        client = app.test_client()
        with client.get("/graph") as response:
            self.assertEqual(response.status_code, 200)
        self.assertEqual(client.get("/api/graph/catalog").get_json()["organs"][0]["count"], 4)
        for query in ("k=0", "k=4", "k=1.5", "branch=-1", "branch=4"):
            self.assertEqual(client.get(f"/api/graph/neighbors/key-c?{query}").status_code, 400)
        self.assertEqual(client.get("/api/graph?projection=unknown").status_code, 400)
        self.assertEqual(
            client.get("/api/graph/neighbors/key-c?k=3").get_json()["indices"], [1, 2, 3]
        )
        response = client.get("/api/graph?projection=umap")
        self.assertIn(response.status_code, (200, 202))
        app.extensions["graph_store"].executor.submit(lambda: None).result(timeout=120)
        response = client.get("/api/graph?projection=umap")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["projection"], "umap")
        self.assertEqual(response.get_json()["umap_neighbors"], 3)


if __name__ == "__main__":
    unittest.main()
