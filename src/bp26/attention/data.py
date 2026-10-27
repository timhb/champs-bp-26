"""Slide selection and complete-bag, branch-specific CLAM-MB attention."""

import hashlib
import os
import threading
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from bp26.inference.champion import load_model
from bp26.inference.manifest import build_manifest


def load_catalog(config):
    """Join model-ready slides to exactly their indexed WSI paths."""
    manifest = Path(config["manifest"])
    if manifest.is_file():
        frame = pd.read_csv(manifest, keep_default_na=False, dtype=str)
        frame["path"] = frame.path.map(
            lambda value: str(Path(value) if Path(value).is_absolute() else manifest.parent / value)
        )
    else:
        frame = build_manifest(config["feature_index"], config["slide_registry"])
    if frame.empty or not set(frame.organ).issubset({"lung", "liver"}):
        raise ValueError("Manifest must contain lung/liver slides")
    if frame.duplicated(["organ", "slide_id"]).any():
        raise ValueError("Manifest has duplicate slide/organ identities")
    index = pd.read_csv(
        config["feature_index"],
        keep_default_na=False,
        dtype=str,
        usecols=[
            "slide_id",
            "path",
            "wsi_path",
            "site",
            "case_id",
            "encoder",
            "resolution",
            "refused",
        ],
    )
    index = index.loc[
        index.encoder.eq("virchow2")
        & index.resolution.eq("20x_224px_0px_overlap")
        & index.refused.eq("")
    ]
    if index.duplicated(["slide_id", "path"]).any():
        raise ValueError("Feature index has duplicate paths")
    frame = frame.merge(index, on=["slide_id", "path"], how="left", validate="one_to_one")
    if frame.wsi_path.isna().any() or frame.wsi_path.eq("").any():
        raise ValueError("Manifest contains features without matching Virchow2 WSI paths")
    root = Path(config["data_root"]).resolve()

    def resolve_wsi(value):
        path = Path(value)
        if not path.is_absolute():
            path = root / value.removeprefix("data/")
        path = Path(os.path.abspath(path))
        if not path.is_relative_to(root):
            raise ValueError("Indexed WSI is outside the configured data root")
        return str(path)

    frame["wsi_path"] = frame.wsi_path.map(resolve_wsi)
    frame["id"] = frame.apply(
        lambda row: hashlib.sha256(f"{row.organ}:{row.slide_id}".encode()).hexdigest()[:20], axis=1
    )
    if frame.id.duplicated().any():
        raise ValueError("Slide identifiers collide")
    return frame.sort_values(["organ", "slide_id"]).set_index("id", drop=False)


def percentiles(weights):
    """Empirical ranks in [0, 1], with tied weights receiving their mean rank."""
    if len(weights) == 1:
        return np.ones(1, dtype=np.float32)
    _, inverse, counts = np.unique(weights, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts + 1) / 2) / (len(weights) - 1)
    return ranks[inverse].astype(np.float32)


def compute_attention(model, feature_path, slide_id, dimensions):
    """Preserve HDF5 row order and full level-0 geometry; never sample the bag."""
    with h5py.File(feature_path, "r") as handle:
        features = np.asarray(handle["features"], dtype=np.float32)
        coords = np.asarray(handle["coords"])
        attrs = dict(handle["coords"].attrs)
        if handle["features"].attrs.get("encoder") != "virchow2":
            raise ValueError("Feature encoder is not Virchow2")
        if attrs.get("name") != slide_id:
            raise ValueError("Feature coordinates belong to a different slide")
    if (
        features.ndim != 2
        or features.shape[1] != 2560
        or len(features) == 0
        or not np.isfinite(features).all()
    ):
        raise ValueError("Invalid Virchow2 feature bag")
    if (
        coords.shape != (len(features), 2)
        or not np.isfinite(coords).all()
        or (coords < 0).any()
        or not np.equal(coords, np.floor(coords)).all()
    ):
        raise ValueError("Invalid or misaligned full-resolution tile coordinates")
    if attrs.get("target_magnification") != 20 or attrs.get("patch_size") != 224:
        raise ValueError("Champion weights require 224px Virchow2 tiles at 20x")
    if (int(attrs["level0_width"]), int(attrs["level0_height"])) != tuple(dimensions):
        raise ValueError("Feature dimensions do not match the whole-slide image")
    patch_size = int(attrs["patch_size_level0"])
    if patch_size <= 0 or (coords >= np.asarray(dimensions)).any():
        raise ValueError("Tile footprint or coordinates exceed slide dimensions")
    with torch.inference_mode():
        output = model(torch.from_numpy(features))
    attention = output["attention"].float().cpu().numpy()
    logits = output["logits"][0].float().cpu().numpy()
    if (
        attention.shape != (len(logits), len(features))
        or not np.isfinite(attention).all()
        or not np.isfinite(logits).all()
        or (attention < 0).any()
        or not np.allclose(attention.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Model returned invalid branch attention")
    return dict(
        coords=coords.astype(np.int64),
        attention=attention,
        logits=logits,
        scores=output["logits"][0].sigmoid().cpu().numpy(),
        patch_size=patch_size,
        dimensions=tuple(dimensions),
    )


class AttentionStore:
    """Bounded in-memory attention cache; frozen models are loaded once per organ."""

    def __init__(self, config, catalog):
        self.config, self.catalog = config, catalog
        self.models, self.cache = {}, OrderedDict()
        self.lock = threading.RLock()

    def get(self, slide_key, dimensions):
        with self.lock:
            if slide_key in self.cache:
                self.cache.move_to_end(slide_key)
                return self.cache[slide_key]
            row = self.catalog.loc[slide_key]
            if row.organ not in self.models:
                root = (
                    Path(self.config["models_root"])
                    / f"focus17_{row.organ}_virchow2_clam_mb_multilabel"
                )
                directories = list(root.glob("*/job_*"))
                if len(directories) != 1:
                    raise ValueError(f"Expected one champion checkpoint directory for {row.organ}")
                self.models[row.organ] = load_model(directories[0], "cpu")
            model, labels, record = self.models[row.organ]
            result = compute_attention(model, row.path, row.slide_id, dimensions)
            if len(labels) != len(result["logits"]):
                raise ValueError("Finding labels and model branches disagree")
            result.update(labels=labels, checkpoint=record)
            self.cache[slide_key] = result
            while len(self.cache) > self.config["attention_cache_size"]:
                self.cache.popitem(last=False)
            return result
