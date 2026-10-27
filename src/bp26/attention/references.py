"""Report-derived reference labels matched by exact WSI path and organ."""

import json
import os
import threading
from pathlib import Path

import pandas as pd


def wsi_key(value, root):
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / value.removeprefix("data/")
    return os.path.abspath(path)


def reference_state(row, label):
    """Missing, masked and nonbinary values must not become negative labels."""
    mask = row.get(f"mask_{label}")
    if mask is not None:
        if str(mask) in {"0", "0.0"}:
            return "masked"
        if str(mask) not in {"1", "1.0"}:
            return "unknown"
    value = str(row.get(f"label_{label}", ""))
    if value in {"1", "1.0"}:
        return "positive"
    if value in {"0", "0.0"}:
        return "negative"
    return "unknown"


class ReferenceFindings:
    def __init__(self, config):
        self.root = Path(config["data_root"])
        self.registry_path = Path(config["slide_registry"])
        self.registry = None
        self.datasets = {}
        self.conditions = None
        self.lock = threading.RLock()

    def registry_findings(self, row):
        if self.registry is None:
            self.registry = {}
            if self.registry_path.is_file():
                frame = pd.read_parquet(
                    self.registry_path, columns=["wsi_path", "organ", "semantic_groups"]
                )
                for record in frame.to_dict("records"):
                    organ = {"left_lung": "lung", "right_lung": "lung"}.get(
                        record["organ"], record["organ"]
                    )
                    key = (organ, wsi_key(record["wsi_path"], self.root))
                    if key[1] is None:
                        continue
                    groups = record["semantic_groups"]
                    values = (
                        {v.strip() for v in groups.split(",") if v.strip()}
                        if isinstance(groups, str)
                        else set()
                    )
                    self.registry.setdefault(key, set()).update(values)
        return sorted(self.registry.get((row.organ, wsi_key(row.wsi_path, self.root)), set()))

    def get(self, row, labels, checkpoint):
        with self.lock:
            return self._get(row, labels, checkpoint)

    def _dataset_rows(self, row, path):
        if path not in self.datasets:
            records = {}
            if path.is_file():
                frame = pd.read_csv(
                    path,
                    dtype=str,
                    keep_default_na=False,
                    usecols=lambda c: (
                        c in {"slide_id", "wsi_path", "split"} or c.startswith(("label_", "mask_"))
                    ),
                )
                for record in frame.to_dict("records"):
                    key = (record["slide_id"], wsi_key(record["wsi_path"], self.root))
                    records.setdefault(key, []).append(record)
            self.datasets[path] = records
        return self.datasets[path].get((row.slide_id, wsi_key(row.wsi_path, self.root)), [])

    def _get(self, row, labels, checkpoint):
        path = Path(checkpoint).resolve().parent.parent / "artifacts/dataset_with_splits.csv"
        return self._reference(row, labels, self._dataset_rows(row, path))

    def _reference(self, row, labels, rows):
        registry = self.registry_findings(row)
        states = {}
        for label in labels:
            values = {reference_state(record, label) for record in rows}
            if len(values) == 1:
                states[label] = values.pop()
            elif len(values) > 1:
                states[label] = "conflicting"
            else:
                states[label] = "positive" if label in registry else "unknown"
        splits = sorted({r["split"] for r in rows if r.get("split")})
        return dict(
            source="Champion reference labels"
            if rows
            else "Slide registry"
            if registry
            else "Unavailable",
            findings=[label for label in labels if states[label] == "positive"]
            if rows
            else registry,
            registry_findings=registry if rows else [],
            states=states,
            split=", ".join(splits),
            note="Report-derived reference labels; not independently verified on each physical slide.",
        )

    def condition_index(self, catalog, models_root):
        """Index positive WSI references only for the current organ model heads."""
        with self.lock:
            if self.conditions is not None:
                return self.conditions
            models = {}
            if models_root is not None:
                for organ in catalog.organ.unique():
                    root = Path(models_root) / f"focus17_{organ}_virchow2_clam_mb_multilabel"
                    directories = list(root.glob("*/job_*"))
                    if len(directories) != 1:
                        continue
                    directory = directories[0]
                    mapping = directory / "artifacts/label_mapping.json"
                    if mapping.is_file():
                        indexed = json.loads(mapping.read_text())["idx_to_label"]
                        labels = [indexed[str(i)] for i in range(len(indexed))]
                        models[organ] = (labels, directory / "artifacts/dataset_with_splits.csv")
            result = {}
            for row in catalog.itertuples():
                if row.organ in models:
                    labels, path = models[row.organ]
                    reference = self._reference(row, labels, self._dataset_rows(row, path))
                    findings = set(reference["findings"]) & set(labels)
                else:
                    findings = set()
                result[row.id] = frozenset(findings)
            self.conditions = result
            return result
