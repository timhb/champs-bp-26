"""Validate raw patch pooling, slide selection and restart identity."""

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from bp26.inference.meanpool import DIMENSION, export, pool_bag, select_slides


class MeanPoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bag = self.root / "features.h5"
        self.values = np.repeat(np.array([1, 2, 3, 4, 10], dtype="f4")[:, None], DIMENSION, axis=1)
        with h5py.File(self.bag, "w") as handle:
            features = handle.create_dataset("features", data=self.values)
            features.attrs["encoder"] = "virchow2"
        self.index = self.root / "index.csv"
        self.row = dict(
            slide_id="S1",
            encoder="virchow2",
            resolution="20x_224px_0px_overlap",
            path="features.h5",
            timestamp="2026-10-01",
            refused="",
            id_conflict="False",
        )
        pd.DataFrame([self.row]).to_csv(self.index, index=False)

    def test_all_patches_equal_weight_with_partial_chunk(self):
        vector, count = pool_bag(self.bag, "cpu", 2)
        np.testing.assert_allclose(vector, self.values.mean(axis=0))
        self.assertEqual(count, 5)

    def test_selection_uses_newest_usable_without_stain_filter(self):
        rows = [
            self.row,
            dict(self.row, timestamp="2026-10-02", path="new.h5"),
            dict(self.row, timestamp="2026-10-03", refused="bad"),
            dict(self.row, slide_id="S2", id_conflict="True"),
            dict(self.row, slide_id="S3_IHC", path="other.h5"),
        ]
        pd.DataFrame(rows).to_csv(self.index, index=False)
        frame = select_slides(self.index, self.row["resolution"])
        self.assertEqual(frame.slide_id.tolist(), ["S1", "S3_IHC"])
        self.assertEqual(frame.path.iloc[0], str(self.root / "new.h5"))

    def test_export_resume_and_changed_index(self):
        out = self.root / "out"
        export(self.index, out, self.row["resolution"], chunk_size=2)
        with h5py.File(out / "slides.h5") as handle:
            np.testing.assert_allclose(handle["embeddings"][0], self.values.mean(axis=0))
            self.assertEqual(handle["completed"][:].tolist(), [True])
            self.assertEqual(handle["n_patches"][:].tolist(), [5])
        # A finished row is skipped on restart, even if its source is temporarily unavailable.
        self.bag.unlink()
        export(self.index, out, self.row["resolution"], chunk_size=2)
        self.assertEqual(json.loads((out / "run.json").read_text())["status"], "complete")
        with self.index.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Resume identity"):
            export(self.index, out, self.row["resolution"])

    def test_invalid_bags_fail_without_marking_complete(self):
        with h5py.File(self.bag, "a") as handle:
            handle["features"][0, 0] = np.nan
        out = self.root / "out"
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            export(self.index, out, self.row["resolution"])
        with h5py.File(out / "slides.h5") as handle:
            self.assertFalse(handle["completed"][0])
        self.assertEqual(json.loads((out / "run.json").read_text())["status"], "failed")
        with h5py.File(self.bag, "a") as handle:
            del handle["features"]
            features = handle.create_dataset("features", shape=(0, DIMENSION), dtype="f4")
            features.attrs["encoder"] = "virchow2"
        with self.assertRaisesRegex(ValueError, "Invalid Virchow2 bag shape"):
            pool_bag(self.bag, "cpu", 2)


if __name__ == "__main__":
    unittest.main()
