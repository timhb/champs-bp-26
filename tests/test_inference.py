"""Check compatibility with the stored champion checkpoint format."""

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from bp26.inference.champion import load_model, pool
from bp26.inference.clam import CLAM_MB
from bp26.inference.manifest import build_manifest
from bp26.inference.predict import predict


class ChampionInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "models/focus17_lung_virchow2_clam_mb_multilabel/fold0/job_1"
        (self.directory / "checkpoints").mkdir(parents=True)
        (self.directory / "artifacts").mkdir()
        self.model = CLAM_MB(2560, 2, hidden_dim=8, attention_dim=4).eval()
        weights = {
            "model."
            + key.replace("projection.", "fc.", 1).replace("attention.", "attention_net.", 1): value
            for key, value in self.model.state_dict().items()
        }
        torch.save(
            {
                "state_dict": weights,
                "epoch": 7,
                "hyper_parameters": {
                    "model_type": "mb",
                    "gate": True,
                    "input_dim": 2560,
                    "num_classes": 2,
                    "hidden_dim": 8,
                    "attention_dim": 4,
                    "dropout": 0.25,
                },
            },
            self.directory / "checkpoints/best.ckpt",
        )
        # last.ckpt is metadata only: it deliberately contains no usable model weights.
        torch.save(
            {
                "callbacks": {
                    "selector": {
                        "best_model_path": "/old/training/path/best.ckpt",
                        "monitor": "val_ap_core_macro",
                        "best_model_score": torch.tensor(0.5),
                    }
                }
            },
            self.directory / "checkpoints/last.ckpt",
        )
        (self.directory / "artifacts/label_mapping.json").write_text(
            json.dumps({"idx_to_label": {"0": "first", "1": "second"}})
        )
        self.features = np.random.default_rng(42).normal(size=(3, 2560)).astype("float32")
        self.bag = self.root / "features.h5"
        with h5py.File(self.bag, "w") as handle:
            handle["features"] = self.features
            coords = handle.create_dataset("coords", data=np.zeros((3, 2), dtype="int64"))
            coords.attrs["target_magnification"] = 20
        self.manifest = self.root / "slides.csv"
        pd.DataFrame([{"organ": "lung", "slide_id": "0001", "path": "features.h5"}]).to_csv(
            self.manifest, index=False
        )

    def test_legacy_checkpoint_selection_and_finding_order(self):
        frame, records = predict(self.manifest, self.root / "models")
        with torch.inference_mode():
            expected = self.model(torch.from_numpy(self.features))["logits"][0]
        np.testing.assert_allclose(frame.logit, expected.numpy(), rtol=1e-6)
        np.testing.assert_allclose(frame.score, expected.sigmoid().numpy(), rtol=1e-6)
        self.assertEqual(frame.finding.tolist(), ["first", "second"])
        self.assertEqual(frame.slide_id.tolist(), ["0001", "0001"])
        self.assertEqual(records["lung"]["epoch"], 7)

    def test_export_pooling_matches_classifier(self):
        model, labels, _ = load_model(self.directory, "cpu")
        bag = torch.from_numpy(self.features)
        embeddings, logits = pool(model, bag)
        self.assertEqual(tuple(embeddings.shape), (len(labels), 8))
        torch.testing.assert_close(logits, model(bag)["logits"][0])

    def test_wrong_magnification_is_rejected(self):
        with h5py.File(self.bag, "r+") as handle:
            handle["coords"].attrs["target_magnification"] = 10
        with self.assertRaisesRegex(ValueError, "expected 20x"):
            predict(self.manifest, self.root / "models")

    def test_nonfinite_features_are_rejected(self):
        with h5py.File(self.bag, "r+") as handle:
            handle["features"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite source features"):
            predict(self.manifest, self.root / "models")

    def test_unknown_organ_is_rejected(self):
        pd.DataFrame([{"organ": "brain", "slide_id": "1", "path": "features.h5"}]).to_csv(
            self.manifest, index=False
        )
        with self.assertRaisesRegex(ValueError, "Organ must be"):
            predict(self.manifest, self.root / "models")

    def test_shared_index_selection(self):
        index = self.root / "feature_index.csv"
        base = dict(
            slide_id="M00001.043",
            encoder="virchow2",
            resolution="20x_224px_0px_overlap",
            path="old.h5",
            timestamp="2026-01-01",
            refused="",
            id_conflict="False",
        )
        pd.DataFrame(
            [
                base,
                {**base, "path": "new.h5", "timestamp": "2026-02-01"},
                {**base, "path": "refused.h5", "timestamp": "2026-03-01", "refused": "empty"},
                {**base, "slide_id": "M00002.041", "path": "liver.h5"},
                {**base, "slide_id": "M00003.043", "id_conflict": "True"},
                {**base, "slide_id": "M00004.043_PAS"},
                {**base, "slide_id": "M00005.043", "encoder": "uni_v2"},
            ]
        ).to_csv(index, index=False)
        registry = self.root / "registry.parquet"
        pd.DataFrame(
            [
                dict(slide_id="M00001.043", organ="right_lung"),
                dict(slide_id="M00001.043", organ="right_lung"),
                dict(slide_id="M00002.041", organ="liver"),
                dict(slide_id="M00003.043", organ="left_lung"),
                dict(slide_id="M00004.043_PAS", organ="left_lung"),
                dict(slide_id="M00005.043", organ="left_lung"),
            ]
        ).to_parquet(registry)
        frame = build_manifest(index, registry)
        self.assertEqual(frame.organ.tolist(), ["liver", "lung"])
        self.assertEqual(frame.path.tolist(), ["liver.h5", "new.h5"])
