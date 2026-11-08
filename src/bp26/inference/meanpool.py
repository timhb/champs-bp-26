"""Export one unsupervised mean of every raw Virchow2 patch bag per WSI."""

import argparse
import json
import platform
import subprocess
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from bp26.inference.champion import sha256

DIMENSION = 2560
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs/embeddings/virchow2_meanpool.toml"


def select_slides(index_path: Path, resolution: str) -> pd.DataFrame:
    """
    Select newest usable indexed bags (Virchow2 and correct resolution), without organ or stain filtering.
    """

    frame = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    required = {"encoder", "resolution", "slide_id", "path", "timestamp", "refused", "id_conflict"}

    if missing := required - set(frame.columns):
        raise ValueError(f"Feature index lacks {sorted(missing)}")

    frame = frame.loc[
        frame.encoder.eq("virchow2")
        & frame.resolution.eq(resolution)
        & frame.refused.eq("")
        & frame.id_conflict.str.lower().isin(["false", "0", ""])
    ].copy()

    if frame.empty or frame.slide_id.eq("").any() or frame.path.eq("").any():
        raise ValueError("No usable slides, or blank slide IDs/paths")

    frame = frame.sort_values(["timestamp", "path"], kind="stable")
    frame = frame.drop_duplicates("slide_id", keep="last").sort_values("slide_id")

    frame["path"] = frame.path.map(
        lambda value: str(
            Path(value) if Path(value).is_absolute() else index_path.resolve().parent / value
        )
    )

    if frame.path.duplicated().any():
        raise ValueError("Multiple slide IDs point to the same feature file")

    return frame.reset_index(drop=True)


@torch.inference_mode()
def pool_bag(path: str | Path, device: str, chunk_size: int) -> tuple[np.ndarray, int]:
    """Bound memory; weight every patch equally, including the last partial chunk."""
    total = torch.zeros(DIMENSION, dtype=torch.float64, device=device)

    with h5py.File(path, "r") as source:
        features = source["features"]

        if features.ndim != 2 or features.shape[1] != DIMENSION or not features.shape[0]:
            raise ValueError(f"Invalid Virchow2 bag shape: {features.shape}")
        if features.attrs.get("encoder") != "virchow2":
            raise ValueError(f"Unexpected feature encoder: {features.attrs.get('encoder')}")

        n_patches = features.shape[0]

        for start in range(0, n_patches, chunk_size):
            batch = torch.from_numpy(np.asarray(features[start : start + chunk_size], dtype="f4"))
            batch = batch.to(device)

            if not torch.isfinite(batch).all():
                raise ValueError(f"Nonfinite patch features: {path}")

            total += batch.sum(dim=0, dtype=torch.float64)

    vector = (total / n_patches).float().cpu().numpy()

    if not np.isfinite(vector).all():
        raise ValueError(f"Nonfinite pooled features: {path}")

    return vector, n_patches


def export(
    index_path: Path,
    out: Path,
    resolution: str,
    device: str = "cpu",
    chunk_size: int = 8192,
    limit: int | None = None,
) -> None:

    if chunk_size <= 0 or (limit is not None and limit <= 0):
        raise ValueError("chunk_size and limit must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    frame = select_slides(index_path, resolution)

    if limit is not None:
        frame = frame.head(limit).copy()

    identity = dict(
        feature_index_sha256=sha256(index_path),
        script_sha256=sha256(__file__),
        resolution=resolution,
        limit=limit,
    )

    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "run.json"
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous["identity"] != identity:
            raise ValueError("Resume identity differs; use a separate output directory")

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=DEFAULT_CONFIG.parents[2],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    record = dict(
        identity=identity,
        status="running",
        feature_index=str(index_path.resolve()),
        encoder="virchow2",
        pooling="Arithmetic mean over all patches; no CLAM or normalization",
        supervised=False,
        dimension=DIMENSION,
        dtype="float32",
        device=device,
        chunk_size=chunk_size,
        code_revision=revision,
        torch_version=torch.__version__,
        host=platform.node(),
        hardware=torch.cuda.get_device_name() if device == "cuda" else "cpu",
        started_at=datetime.now(timezone.utc).isoformat(),
        total=len(frame),
    )

    record_path.write_text(json.dumps(record, indent=2) + "\n")

    start = time.monotonic()
    try:
        with h5py.File(out / "slides.h5", "a") as dest:
            if "embeddings" not in dest:
                dest.attrs["identity"] = json.dumps(identity, sort_keys=True)
                dest.attrs.update(encoder="virchow2", pooling="mean", resolution=resolution)
                dest.create_dataset(
                    "slide_ids",
                    data=frame.slide_id.to_numpy(dtype=object),
                    dtype=h5py.string_dtype(),
                )
                dest.create_dataset(
                    "embeddings",
                    (len(frame), DIMENSION),
                    dtype="f4",
                    chunks=(1, DIMENSION),
                    fillvalue=np.nan,
                )
                dest.create_dataset("n_patches", data=np.zeros(len(frame), dtype="i8"))
                dest.create_dataset("completed", data=np.zeros(len(frame), dtype=bool))
                dest.flush()

            if (
                dest.attrs["identity"] != json.dumps(identity, sort_keys=True)
                or dest["slide_ids"].asstr()[:].tolist() != frame.slide_id.tolist()
                or dest["embeddings"].shape != (len(frame), DIMENSION)
            ):
                raise ValueError("Existing HDF5 identity differs")

            frame["embedding_row"] = np.arange(len(frame))
            frame["embedding_file"] = str((out / "slides.h5").resolve())
            frame.to_csv(out / "slides.csv", index=False)
            for row in frame.loc[~dest["completed"][:]].itertuples():
                vector, count = pool_bag(row.path, device, chunk_size)
                dest["embeddings"][row.Index] = vector
                dest["n_patches"][row.Index] = count
                dest.flush()
                dest["completed"][row.Index] = True
                dest.flush()
                if row.Index % 100 == 0:
                    print(
                        f"{row.Index + 1}/{len(frame)} slides; {time.monotonic() - start:.1f}s",
                        flush=True,
                    )
            record["completed"] = int(dest["completed"][:].sum())
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Complete: {record['completed']} WSI vectors in {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--feature-index", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--limit", type=int, help="Smoke test; use a separate output directory")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    torch.set_num_threads(1)
    export(
        args.feature_index or Path(config["feature_index"]),
        args.out or Path(config["out"]),
        config["resolution"],
        args.device,
        args.chunk_size if args.chunk_size is not None else config["chunk_size"],
        args.limit,
    )


if __name__ == "__main__":
    main()
