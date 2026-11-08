"""Predict lung/liver findings from Virchow2 HDF5 bags.

The CSV manifest requires organ (lung/liver), slide_id and path (feature HDF5).
Relative feature paths resolve against the manifest's directory. Output is one
row per slide and finding, with the raw logit and sigmoid score. These are
inference scores; the manifest may include training slides.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import torch

from bp26.inference.champion import load_model, read_bag
from bp26.inference.settings import settings


def predict(manifest, models_root, device="cpu"):
    """Score each complete feature bag with its organ's selected checkpoint."""
    frame = pd.read_csv(manifest, keep_default_na=False, dtype=str)
    required = ["organ", "slide_id", "path"]
    if frame.empty or not set(required).issubset(frame.columns):
        raise ValueError("Manifest needs rows and organ, slide_id, path columns")
    if frame[required].eq("").any().any() or frame.duplicated(["organ", "slide_id"]).any():
        raise ValueError("Missing manifest values or duplicate slide/organ identities")
    if not set(frame.organ).issubset({"lung", "liver"}):
        raise ValueError("Organ must be lung or liver")
    predictions, records = [], {}
    for organ, rows in frame.groupby("organ", sort=False):
        root = Path(models_root) / f"focus17_{organ}_virchow2_clam_mb_multilabel"
        directories = list(root.glob("*/job_*"))
        if len(directories) != 1:
            raise ValueError(f"Expected exactly one champion model directory for {organ}: {root}")
        model, labels, records[organ] = load_model(directories[0], device)
        for row in rows.itertuples():
            path = Path(row.path)
            if not path.is_absolute():
                path = Path(manifest).resolve().parent / path
            features, magnification, error = read_bag(path)
            if error:
                raise ValueError(f"{row.slide_id}: {error}")
            if magnification != 20:
                raise ValueError(f"{row.slide_id}: expected 20x Virchow2 features")
            with torch.inference_mode():
                bag = torch.from_numpy(features).to(device)
                logits = model(bag)["logits"].squeeze(0).float().cpu()
                if not torch.isfinite(logits).all():
                    raise ValueError(f"{row.slide_id}: nonfinite model output")
                scores = logits.sigmoid().tolist()
            for label, logit, score in zip(labels, logits.tolist(), scores, strict=True):
                predictions.append(
                    dict(
                        organ=organ, slide_id=row.slide_id, finding=label, logit=logit, score=score
                    )
                )
    return pd.DataFrame(predictions), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Prediction CSV")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(
            f"Manifest not found: {args.manifest}. Build it with: "
            "pixi run python scripts/inference/build_manifest.py --out data/inference/slides.csv"
        )
    torch.set_num_threads(4)
    frame, models = predict(
        args.manifest, args.models_root or settings(args.config)["models_root"], args.device
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=args.out.parent, suffix=".csv")
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, args.out)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(json.dumps({"output": str(args.out), "rows": len(frame), "models": models}))
