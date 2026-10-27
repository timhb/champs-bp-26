"""Reference labels must match the exact WSI, preserve masks, and expose missingness."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from bp26.attention.references import ReferenceFindings, reference_state


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = self.root / "registry.parquet"
        self.checkpoint = self.root / "models/job/checkpoints/best.ckpt"
        self.dataset = self.checkpoint.parent.parent / "artifacts/dataset_with_splits.csv"
        self.dataset.parent.mkdir(parents=True)
        self.row = SimpleNamespace(
            organ="lung", slide_id="slide", wsi_path=str(self.root / "site/slide.svs")
        )
        self.config = dict(data_root=self.root, slide_registry=self.registry)
        pd.DataFrame(
            [
                dict(
                    wsi_path="data/site/slide.svs",
                    organ="right_lung",
                    semantic_groups="finding_b,other_finding",
                ),
                dict(
                    wsi_path="data/site/slide.svs", organ="right_lung", semantic_groups="finding_b"
                ),
                dict(
                    wsi_path="data/other/slide.svs", organ="right_lung", semantic_groups="finding_a"
                ),
            ]
        ).to_parquet(self.registry)

    def test_binary_masks_and_unknown_values(self):
        self.assertEqual(reference_state({"label_a": "1.0"}, "a"), "positive")
        self.assertEqual(reference_state({"label_a": "0"}, "a"), "negative")
        self.assertEqual(reference_state({}, "a"), "unknown")
        self.assertEqual(reference_state({"label_a": "0.5"}, "a"), "unknown")
        self.assertEqual(reference_state({"label_a": "1", "mask_a": "0"}, "a"), "masked")
        self.assertEqual(reference_state({"label_a": "1", "mask_a": ""}, "a"), "unknown")

    def test_champion_labels_preferred_over_other_registry_run(self):
        pd.DataFrame(
            [
                dict(
                    slide_id="slide",
                    wsi_path="data/site/slide.svs",
                    split="train",
                    label_finding_a="1",
                    label_finding_b="0",
                ),
                dict(
                    slide_id="slide",
                    wsi_path="data/other/slide.svs",
                    split="test",
                    label_finding_a="0",
                    label_finding_b="1",
                ),
            ]
        ).to_csv(self.dataset, index=False)
        result = ReferenceFindings(self.config).get(
            self.row, ["finding_a", "finding_b", "missing"], self.checkpoint
        )
        self.assertEqual(result["findings"], ["finding_a"])
        self.assertEqual(
            result["states"], dict(finding_a="positive", finding_b="negative", missing="unknown")
        )
        self.assertEqual(result["split"], "train")
        self.assertEqual(result["registry_findings"], ["finding_b", "other_finding"])

    def test_missing_champion_row_only_uses_registry_positives(self):
        result = ReferenceFindings(self.config).get(
            self.row, ["finding_a", "finding_b"], self.checkpoint
        )
        self.assertEqual(result["source"], "Slide registry")
        self.assertEqual(result["states"], dict(finding_a="unknown", finding_b="positive"))
        self.assertEqual(result["findings"], ["finding_b", "other_finding"])

    def test_conflicting_champion_rows_not_arbitrarily_selected(self):
        pd.DataFrame(
            [
                dict(slide_id="slide", wsi_path="data/site/slide.svs", label_finding_a="1"),
                dict(slide_id="slide", wsi_path="data/site/slide.svs", label_finding_a="0"),
            ]
        ).to_csv(self.dataset, index=False)
        result = ReferenceFindings(self.config).get(self.row, ["finding_a"], self.checkpoint)
        self.assertEqual(result["states"]["finding_a"], "conflicting")
        self.assertEqual(result["findings"], [])

    def test_condition_index_matches_displayed_references(self):
        directory = (
            self.root / "models/focus17_lung_virchow2_clam_mb_multilabel/fold0/job_1/artifacts"
        )
        directory.mkdir(parents=True)
        (directory / "label_mapping.json").write_text(
            json.dumps({"idx_to_label": {"0": "finding_a", "1": "finding_b"}})
        )
        pd.DataFrame(
            [
                dict(
                    slide_id="slide",
                    wsi_path="data/site/slide.svs",
                    label_finding_a="1",
                    mask_finding_a="1",
                    label_finding_b="0",
                ),
                dict(
                    slide_id="masked",
                    wsi_path="data/site/masked.svs",
                    label_finding_a="1",
                    mask_finding_a="0",
                    label_finding_b="",
                ),
            ]
        ).to_csv(directory / "dataset_with_splits.csv", index=False)
        catalog = pd.DataFrame(
            [
                dict(
                    id="primary",
                    organ="lung",
                    slide_id="slide",
                    wsi_path=str(self.root / "site/slide.svs"),
                ),
                dict(
                    id="fallback",
                    organ="lung",
                    slide_id="other",
                    wsi_path=str(self.root / "other/slide.svs"),
                ),
                dict(
                    id="masked",
                    organ="lung",
                    slide_id="masked",
                    wsi_path=str(self.root / "site/masked.svs"),
                ),
            ]
        )
        registry = pd.read_parquet(self.registry)
        registry.loc[registry.wsi_path.eq("data/other/slide.svs"), "semantic_groups"] = (
            "finding_a,unpredicted_condition"
        )
        registry.to_parquet(self.registry)
        references = ReferenceFindings(self.config)
        indexed = references.condition_index(catalog, self.root / "models")
        self.assertEqual(indexed["primary"], frozenset({"finding_a"}))
        self.assertNotIn("finding_b", indexed["primary"])
        self.assertEqual(indexed["fallback"], frozenset({"finding_a"}))
        self.assertNotIn("unpredicted_condition", indexed["fallback"])
        self.assertEqual(indexed["masked"], frozenset())
        displayed = references.get(
            self.row, ["finding_a", "finding_b"], directory.parent / "checkpoints/best.ckpt"
        )
        self.assertEqual(indexed["primary"], frozenset(displayed["findings"]))
