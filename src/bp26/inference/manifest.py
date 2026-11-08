"""Build an inference manifest from the shared feature index and slide registry."""

import argparse
import json
from pathlib import Path

import pandas as pd

from bp26.inference.champion import sha256
from bp26.inference.settings import settings
from bp26.inference.stains import is_he_slide


def build_manifest(index_path, registry_path):
    """Use registry organs and the source pipeline's newest usable H&E features."""
    index = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    required = {"slide_id", "encoder", "resolution", "path", "timestamp", "refused", "id_conflict"}
    if not required.issubset(index.columns):
        raise ValueError(f"Feature index lacks columns: {sorted(required - set(index.columns))}")
    usable = index.loc[
        index.encoder.eq("virchow2")
        & index.resolution.eq("20x_224px_0px_overlap")
        & index.refused.eq("")
        & index.id_conflict.str.lower().isin(["false", "0", ""])
        & index.slide_id.map(is_he_slide)
    ].copy()
    usable = usable.sort_values("timestamp", kind="stable").drop_duplicates("slide_id", keep="last")
    registry = pd.read_parquet(registry_path, columns=["slide_id", "organ"])
    registry["organ"] = registry.organ.replace({"left_lung": "lung", "right_lung": "lung"})
    registry = registry[registry.organ.isin(["lung", "liver"])].drop_duplicates()
    if registry.slide_id.duplicated().any():
        raise ValueError("Registry assigns conflicting organs to a slide")
    result = usable.merge(registry, on="slide_id", how="inner", validate="one_to_one")
    if result.empty:
        raise ValueError("No usable lung/liver Virchow2 features found")
    return result[["organ", "slide_id", "path"]].sort_values(["organ", "slide_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--feature-index", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = settings(args.config)
    index = args.feature_index or Path(config["feature_index"])
    registry = args.registry or Path(config["slide_registry"])
    frame = build_manifest(index, registry)
    # Resolve source-relative paths before writing a manifest in another directory.
    frame["path"] = frame.path.map(
        lambda value: str(
            Path(value) if Path(value).is_absolute() else index.resolve().parent / value
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    record = {
        "feature_index": str(index),
        "feature_index_sha256": sha256(index),
        "registry": str(registry),
        "registry_sha256": sha256(registry),
        "manifest_sha256": sha256(args.out),
        "slides_by_organ": frame.organ.value_counts().to_dict(),
        "selection": "Newest non-refused 20x 224px Virchow2 features; H&E name filter; no identifier conflicts; registry lung/liver organs",
    }
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.out), "slides_by_organ": record["slides_by_organ"]}))
