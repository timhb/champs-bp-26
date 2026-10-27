"""Export all class-specific CLAM MB slide representations from stored Virchow2 bags.

Each organ HDF5 contains embeddings (slides, finding branches, 512), logits,
slide_ids, branch_labels, and a completed mask. Rows match slides.csv. These
are supervised representations from the selected checkpoint, not held-out
predictions. Input patch magnification is preserved, never relabelled.
"""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from bp26.inference.champion import load_model, pool, read_bag, sha256
from bp26.inference.settings import settings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--models-root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--read-workers", type=int, default=4)
    ap.add_argument(
        "--limit", type=int, default=None, help="Smoke-test only; use a separate output directory"
    )
    args = ap.parse_args()
    args.models_root = args.models_root or Path(settings(args.config)["models_root"])
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit must be positive")
    if args.read_workers <= 0:
        ap.error("--read-workers must be positive")
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest, keep_default_na=False, dtype={"slide_id": str})
    if frame.empty or not {"organ", "slide_id", "path"}.issubset(frame.columns):
        raise ValueError("Manifest needs rows and organ, slide_id, path columns")
    if not set(frame.organ).issubset({"lung", "liver"}):
        raise ValueError("Organ must be lung or liver")
    frame["path"] = frame["path"].map(
        lambda value: str(
            Path(value) if Path(value).is_absolute() else args.manifest.resolve().parent / value
        )
    )
    if frame.duplicated(["organ", "slide_id"]).any():
        raise ValueError("Duplicate slide/organ identities")
    record_path = args.out / "run.json"
    record = dict(
        status="running",
        input_manifest=str(args.manifest.resolve()),
        manifest_sha256=sha256(args.manifest),
        script_sha256=sha256(__file__),
        encoder="virchow2",
        source_magnification=20,
        representation="One attention-pooled hidden vector per finding branch; no branch averaging",
        supervised=True,
        evaluation="Full corpus export; includes training slides",
        dtype="float32",
        device=args.device,
        started_at=datetime.now(timezone.utc).isoformat(),
        organs={},
    )
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        for key in ("manifest_sha256", "script_sha256"):
            if previous[key] != record[key]:
                raise ValueError(f"Resume identity mismatch: {key}")
    record_path.write_text(json.dumps(record, indent=2))
    start = time.monotonic()
    for organ in ("lung", "liver"):
        rows = frame[frame.organ.eq(organ)].copy().reset_index(drop=True)
        if args.limit:
            rows = rows.head(args.limit)
        if rows.empty:
            continue
        root = args.models_root / f"focus17_{organ}_virchow2_clam_mb_multilabel"
        directories = list(root.glob("*/job_*"))
        if len(directories) != 1:
            raise ValueError(f"Ambiguous model directories for {organ}")
        model, labels, model_record = load_model(directories[0], args.device)
        record["organs"][organ] = model_record
        rows["embedding_row"] = np.arange(len(rows))
        rows["embedding_file"] = str((args.out / f"{organ}.h5").resolve())
        rows.to_csv(args.out / f"{organ}_slides.csv", index=False)
        with h5py.File(args.out / f"{organ}.h5", "a") as dest:
            if "embeddings" not in dest:
                dest.create_dataset(
                    "embeddings",
                    (len(rows), len(labels), 512),
                    dtype="f4",
                    chunks=(1, len(labels), 512),
                    fillvalue=np.nan,
                )
                dest.create_dataset(
                    "logits", (len(rows), len(labels)), dtype="f4", fillvalue=np.nan
                )
                dest.create_dataset("completed", data=np.zeros(len(rows), dtype=bool))
                dest.create_dataset(
                    "slide_ids",
                    data=rows.slide_id.to_numpy(dtype=object),
                    dtype=h5py.string_dtype(),
                )
                dest.create_dataset("branch_labels", data=labels, dtype=h5py.string_dtype())
                dest.attrs.update(
                    manifest_sha256=record["manifest_sha256"],
                    checkpoint_sha256=model_record["sha256"],
                    source_magnification=20,
                    encoder="virchow2",
                    organ=organ,
                )
            if (
                dest.attrs["manifest_sha256"] != record["manifest_sha256"]
                or dest.attrs["checkpoint_sha256"] != model_record["sha256"]
                or dest["embeddings"].shape != (len(rows), len(labels), 512)
                or dest["slide_ids"].asstr()[:].tolist() != rows.slide_id.tolist()
            ):
                raise ValueError("Existing output identity differs")
            todo = rows.loc[~dest["completed"][:]]
            errors = []
            # Separate processes avoid h5py's global lock serializing file reads.
            # A fresh process context also avoids inheriting initialized CUDA state.
            with ProcessPoolExecutor(
                max_workers=args.read_workers, mp_context=multiprocessing.get_context("spawn")
            ) as executor:
                iterator = iter(todo.itertuples())
                pending = []
                for _ in range(args.read_workers):
                    row = next(iterator, None)
                    if row is not None:
                        pending.append((row, executor.submit(read_bag, row.path)))
                while pending:
                    row, future = pending.pop(0)
                    features, mag, error = future.result()
                    next_row = next(iterator, None)
                    if next_row is not None:
                        pending.append((next_row, executor.submit(read_bag, next_row.path)))
                    if error:
                        errors.append(dict(slide_id=row.slide_id, path=row.path, error=error))
                        print(f"ERROR {row.slide_id}: {error}", flush=True)
                        continue
                    if mag != 20:
                        raise ValueError(f"Unexpected source magnification {mag}: {row.path}")
                    bag = torch.from_numpy(features).to(args.device)
                    pooled, logits = pool(model, bag)
                    if not torch.isfinite(pooled).all() or not torch.isfinite(logits).all():
                        raise ValueError(f"Nonfinite output for {row.slide_id}")
                    dest["embeddings"][row.Index] = pooled.cpu().numpy()
                    dest["logits"][row.Index] = logits.cpu().numpy()
                    dest.flush()
                    dest["completed"][row.Index] = True
                    dest.flush()
                    if row.Index % 100 == 0:
                        print(
                            f"{organ}: row {row.Index + 1}/{len(rows)}, elapsed {time.monotonic() - start:.1f}s",
                            flush=True,
                        )
                    del bag, features, pooled, logits
            completed = int(dest["completed"][:].sum())
            model_record.update(
                total=len(rows),
                completed=completed,
                errors=errors,
                shape=list(dest["embeddings"].shape),
            )
            print(f"{organ}: {completed}/{len(rows)} complete", flush=True)
        record_path.write_text(json.dumps(record, indent=2))
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    record["status"] = (
        "complete"
        if all(v["completed"] == v["total"] for v in record["organs"].values())
        else "incomplete"
    )
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    record["elapsed_seconds"] = time.monotonic() - start
    record_path.write_text(json.dumps(record, indent=2))
    print(record["status"], flush=True)
