"""Attention normalization, full-resolution alignment, and local API contracts."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image

from bp26.attention.data import compute_attention, load_catalog, percentiles
from bp26.attention.server import create_app
from bp26.inference.clam import CLAM_MB


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.features = np.random.default_rng(42).normal(size=(3, 2560)).astype("float32")
        self.coords = np.array([[0, 0], [448, 0], [0, 448]])
        self.path = self.root / "features.h5"
        with h5py.File(self.path, "w") as handle:
            f = handle.create_dataset("features", data=self.features)
            f.attrs["encoder"] = "virchow2"
            c = handle.create_dataset("coords", data=self.coords)
            c.attrs.update(
                name="slide",
                target_magnification=20,
                patch_size=224,
                patch_size_level0=448,
                level0_width=1000,
                level0_height=1000,
            )
        self.model = CLAM_MB(2560, 2, hidden_dim=8, attention_dim=4).eval()
        self.config = dict(
            data_root=self.root,
            feature_index=self.root / "index.csv",
            manifest=self.root / "slides.csv",
            slide_registry=self.root / "registry.parquet",
            models_root=self.root,
            attention_cache_size=2,
            open_slide_cache_size=2,
        )
        pd.DataFrame([dict(organ="lung", slide_id="slide", path=str(self.path))]).to_csv(
            self.config["manifest"], index=False
        )
        pd.DataFrame(
            [
                dict(
                    slide_id="slide",
                    path=str(self.path),
                    wsi_path="data/country/slide.svs",
                    site="country",
                    case_id="case1",
                    encoder="virchow2",
                    resolution="20x_224px_0px_overlap",
                    refused="",
                )
            ]
        ).to_csv(self.config["feature_index"], index=False)

    def test_full_bag_attention_and_level_zero_footprint(self):
        result = compute_attention(self.model, self.path, "slide", (1000, 1000))
        with torch.inference_mode():
            expected = self.model(torch.from_numpy(self.features))
        np.testing.assert_allclose(result["attention"], expected["attention"].numpy())
        np.testing.assert_allclose(result["attention"].sum(axis=1), [1, 1], atol=1e-6)
        np.testing.assert_array_equal(result["coords"], self.coords)
        self.assertEqual(result["patch_size"], 448)
        self.assertEqual(result["attention"].shape, (2, 3))

    def test_tied_percentiles(self):
        np.testing.assert_allclose(percentiles(np.array([0.1, 0.1, 0.8])), [0.25, 0.25, 1])
        np.testing.assert_allclose(percentiles(np.array([0.5, 0.5])), [0.5, 0.5])
        np.testing.assert_allclose(percentiles(np.array([1])), [1])

    def test_wrong_slide_or_dimensions_rejected(self):
        with self.assertRaisesRegex(ValueError, "different slide"):
            compute_attention(self.model, self.path, "other", (1000, 1000))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            compute_attention(self.model, self.path, "slide", (500, 1000))
        with h5py.File(self.path, "r+") as handle:
            handle["features"].attrs["encoder"] = "uni_v2"
        with self.assertRaisesRegex(ValueError, "not Virchow2"):
            compute_attention(self.model, self.path, "slide", (1000, 1000))

    def test_registry_paths_rebased_and_outside_root_rejected(self):
        catalog = load_catalog(self.config)
        self.assertEqual(catalog.iloc[0].wsi_path, str(self.root / "country/slide.svs"))
        frame = pd.read_csv(self.config["feature_index"])
        frame["wsi_path"] = "/outside/slide.svs"
        frame.to_csv(self.config["feature_index"], index=False)
        with self.assertRaisesRegex(ValueError, "outside"):
            load_catalog(self.config)

    def test_catalog_normalizes_stain_names(self):
        manifest = pd.read_csv(self.config["manifest"])
        index = pd.read_csv(self.config["feature_index"], keep_default_na=False)
        for slide_id, expected in [("slide", "HE"), ("slide_GM", "GRAM"), ("slide_PAS", "PAS")]:
            manifest["slide_id"] = slide_id
            index["slide_id"] = slide_id
            manifest.to_csv(self.config["manifest"], index=False)
            index.to_csv(self.config["feature_index"], index=False)
            self.assertEqual(load_catalog(self.config).iloc[0].stain, expected)

    def test_api_branches_attention_export_and_tiles(self):
        catalog = load_catalog(self.config)
        key = catalog.iloc[0].id
        result = compute_attention(self.model, self.path, "slide", (1000, 1000))
        result.update(
            labels=["finding_a", "finding_b"], checkpoint=dict(checkpoint="best.ckpt", epoch=7)
        )
        app = create_app(self.config, catalog)
        app.extensions["attention_store"].cache[key] = result

        class FakeSlide:
            dimensions = (1000, 1000)
            properties = {"openslide.objective-power": "40", "openslide.mpp-x": "0.25"}

        class FakeZoom:
            def get_dzi(self, fmt):
                return '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" TileSize="254" Overlap="1" Format="jpeg"><Size Width="1000" Height="1000"/></Image>'

            def get_tile(self, level, address):
                return Image.new("RGB", (254, 254), "white")

        with patch("bp26.attention.server.SlideCache.get", return_value=(FakeSlide(), FakeZoom())):
            client = app.test_client()
            self.assertEqual(client.get("/api/slides?organ=lung&q=case1").get_json()["total"], 1)
            detail = client.get(f"/api/slides/{key}").get_json()
            self.assertEqual([b["label"] for b in detail["branches"]], result["labels"])
            self.assertEqual(detail["objective"], 40)
            self.assertEqual(detail["mpp_x"], 0.25)
            self.assertEqual(detail["branches"][0]["reference"], "unknown")
            heat = client.get(f"/api/slides/{key}/attention/1").get_json()
            np.testing.assert_allclose(heat["weights"], result["attention"][1])
            self.assertEqual(heat["patch_size"], 448)
            exported = client.get(f"/api/slides/{key}/export/1.csv")
            self.assertIn("finding_b", exported.text)
            self.assertEqual(len(exported.text.splitlines()), 4)
            self.assertEqual(client.get(f"/api/slides/{key}/attention/9").status_code, 404)
            self.assertEqual(client.get("/api/slides/unknown").status_code, 404)
            self.assertEqual(client.get("/api/slides?offset=wrong").status_code, 400)
            self.assertIn(b"<Image", client.get(f"/dzi/{key}.dzi").data)
            self.assertEqual(client.get(f"/dzi/{key}_files/10/0_0.jpeg").mimetype, "image/jpeg")

    def test_conditions_filter_individual_wsi_and_combine_with_other_filters(self):
        first = load_catalog(self.config).iloc[0].to_dict()
        rows = [
            first | dict(id="positive", slide_id="target-one"),
            first | dict(id="negative", slide_id="target-two", stain="PAS"),
            first | dict(id="other_site", slide_id="target-three", site="other"),
            first | dict(id="other_organ", slide_id="target-four", organ="liver"),
        ]
        catalog = pd.DataFrame(rows).set_index("id", drop=False)
        app = create_app(self.config, catalog)
        references = app.extensions["reference_findings"]
        references.conditions = dict(
            positive=frozenset({"finding_a"}),
            negative=frozenset(),
            other_site=frozenset({"finding_a"}),
            other_organ=frozenset({"finding_a"}),
        )
        client = app.test_client()
        result = client.get(
            "/api/slides?organ=lung&site=country&q=target&condition=finding_a"
        ).get_json()
        self.assertEqual(result["total"], 1)
        self.assertEqual([row["id"] for row in result["slides"]], ["positive"])
        self.assertEqual(result["conditions"], [dict(value="finding_a", count=1)])
        self.assertEqual(client.get("/api/slides?organ=lung&site=country").get_json()["total"], 2)
        self.assertEqual(result["stains"], ["HE", "PAS"])
        self.assertEqual(result["slides"][0]["stain"], "HE")
        stained = client.get("/api/slides?organ=lung&site=country&stain=PAS").get_json()
        self.assertEqual([row["id"] for row in stained["slides"]], ["negative"])
        self.assertEqual(stained["conditions"], [dict(value="finding_a", count=0)])
        combined = client.get("/api/slides?stain=PAS&condition=finding_a").get_json()
        self.assertEqual(combined["total"], 0)
        self.assertEqual(client.get("/api/slides?stain=missing").get_json()["total"], 0)
        self.assertEqual(client.get("/api/slides?condition=missing").get_json()["total"], 0)
        self.assertEqual(
            client.get("/api/slides?condition=finding_a&offset=1").get_json()["total"], 3
        )

    def test_filter_options_are_scoped_to_organ(self):
        first = load_catalog(self.config).iloc[0].to_dict()
        catalog = pd.DataFrame(
            [
                first | dict(id="lung_he", stain="HE"),
                first | dict(id="lung_gram", stain="GRAM"),
                first | dict(id="liver_pas", organ="liver", stain="PAS"),
            ]
        ).set_index("id", drop=False)
        app = create_app(self.config, catalog)
        app.extensions["reference_findings"].conditions = dict(
            lung_he=frozenset({"lung_finding"}),
            lung_gram=frozenset(),
            liver_pas=frozenset({"liver_finding"}),
        )
        client = app.test_client()
        for organ, stains, finding in [
            ("lung", ["GRAM", "HE"], "lung_finding"),
            ("liver", ["PAS"], "liver_finding"),
        ]:
            result = client.get(f"/api/slides?organ={organ}").get_json()
            self.assertEqual(result["stains"], stains)
            self.assertEqual(result["conditions"], [dict(value=finding, count=1)])
            empty = client.get(f"/api/slides?organ={organ}&q=missing").get_json()
            self.assertEqual(empty["total"], 0)
            self.assertEqual(empty["stains"], stains)
            self.assertEqual(empty["conditions"], [dict(value=finding, count=0)])
        all_organs = client.get("/api/slides").get_json()
        self.assertEqual(all_organs["stains"], ["GRAM", "HE", "PAS"])
        self.assertEqual(
            [item["value"] for item in all_organs["conditions"]],
            ["liver_finding", "lung_finding"],
        )
        unknown = client.get("/api/slides?organ=unknown").get_json()
        self.assertEqual(unknown["stains"], [])
        self.assertEqual(unknown["conditions"], [])

    def test_registry_adds_image_only_stains_without_feature_files(self):
        pd.DataFrame(
            [
                dict(
                    slide_id="slide_GM",
                    organ="right_lung",
                    wsi_path="data/country/Cases/case1/slide_GM.svs",
                ),
                dict(
                    slide_id="slide_PAS",
                    organ="liver",
                    wsi_path="data/country/Cases/case1/slide_PAS.svs",
                ),
                dict(
                    slide_id="other_HE",
                    organ="lung",
                    wsi_path="data/country/Cases/case1/other_HE.svs",
                ),
            ]
        ).to_parquet(self.config["slide_registry"])
        catalog = load_catalog(self.config)
        self.assertEqual(set(catalog.stain), {"HE", "GRAM", "PAS"})
        self.assertNotIn("other_HE", set(catalog.slide_id))
        gram = catalog.loc[catalog.stain.eq("GRAM")].iloc[0]
        self.assertEqual(gram.path, "")
        self.assertEqual(gram.case_id, "case1")
        self.assertEqual(gram.site, "country")
        self.assertEqual(gram.organ, "lung")
        self.assertEqual(gram.wsi_path, str(self.root / "country/Cases/case1/slide_GM.svs"))

    def test_non_he_images_never_request_model_attention(self):
        catalog = load_catalog(self.config)
        catalog["stain"] = "GRAM"
        app = create_app(self.config, catalog)
        key = catalog.iloc[0].id

        class FakeSlide:
            dimensions = (1000, 2000)
            properties = {"openslide.objective-power": "40"}

            def get_thumbnail(self, size):
                return Image.new("RGB", size, "white")

        class FakeZoom:
            def get_dzi(self, fmt):
                return "<Image/>"

            def get_tile(self, level, address):
                return Image.new("RGB", (254, 254), "white")

        with (
            patch("bp26.attention.server.SlideCache.get", return_value=(FakeSlide(), FakeZoom())),
            patch(
                "bp26.attention.server.AttentionStore.get",
                side_effect=AssertionError("No inference"),
            ) as inference,
        ):
            client = app.test_client()
            response = client.get(f"/api/slides/{key}")
            self.assertEqual(response.status_code, 200)
            detail = response.get_json()
            self.assertFalse(detail["attention_available"])
            self.assertEqual(detail["stain"], "GRAM")
            self.assertEqual(detail["dimensions"], [1000, 2000])
            self.assertEqual(detail["branches"], [])
            thumbnail = client.get(f"/api/slides/{key}/thumbnail.jpg")
            self.assertEqual(thumbnail.mimetype, "image/jpeg")
            self.assertEqual(Image.open(io.BytesIO(thumbnail.data)).size, (176, 128))
            self.assertEqual(client.get("/api/slides/unknown/thumbnail.jpg").status_code, 404)
            self.assertEqual(client.get(f"/dzi/{key}.dzi").status_code, 200)
            self.assertEqual(client.get(f"/dzi/{key}_files/10/0_0.jpeg").mimetype, "image/jpeg")
            for suffix in ["attention/0", "export/0.csv", "patch/0.jpg"]:
                response = client.get(f"/api/slides/{key}/{suffix}")
                self.assertEqual(response.status_code, 404)
                self.assertIn("non-H&E", response.get_json()["error"])
            inference.assert_not_called()
        self.assertTrue(app.extensions["graph_store"].catalog.empty)
