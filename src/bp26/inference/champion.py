"""Load the stored champion checkpoints and their original finding order.

Loader, bag validation and pooling copied from the source's uncommitted
scripts/slides/pool_clam_embeddings.py (workspace snapshot).
"""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from bp26.inference.clam import CLAM_MB


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(directory, device):
    # The saved callback records the validation-selected checkpoint; last.ckpt
    # is used only to read this record, never as a model selection criterion.
    last = torch.load(directory / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
    callbacks = [v for v in last["callbacks"].values() if "best_model_path" in v]
    if len(callbacks) != 1:
        raise ValueError("Expected one validation checkpoint selector")
    selected = directory / "checkpoints" / Path(callbacks[0]["best_model_path"]).name
    state = torch.load(selected, map_location="cpu", weights_only=False)
    hp = state["hyper_parameters"]
    if hp["model_type"] != "mb" or not hp["gate"] or hp["input_dim"] != 2560:
        raise ValueError("Expected gated CLAM MB on 2560-dimensional Virchow2 features")
    model = CLAM_MB(
        input_dim=hp["input_dim"],
        n_classes=hp["num_classes"],
        hidden_dim=hp["hidden_dim"],
        attention_dim=hp["attention_dim"],
        dropout=hp["dropout"],
    )
    weights = {
        k.removeprefix("model.")
        .replace("fc.", "projection.", 1)
        .replace("attention_net.", "attention.", 1): v
        for k, v in state["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False).eval().to(device)
    mapping = json.loads((directory / "artifacts/label_mapping.json").read_text())["idx_to_label"]
    labels = [mapping[str(i)] for i in range(hp["num_classes"])]
    record = dict(
        checkpoint=str(selected),
        sha256=sha256(selected),
        epoch=state["epoch"],
        selection_monitor=callbacks[0]["monitor"],
        selection_score=float(callbacks[0]["best_model_score"]),
        labels=labels,
    )
    return model, labels, record


def read_bag(path):
    try:
        with h5py.File(path, "r") as handle:
            features = np.asarray(handle["features"], dtype=np.float32)
            attrs = dict(handle["coords"].attrs)
            mag = float(attrs["target_magnification"])
            if handle["coords"].shape != (len(features), 2):
                raise ValueError("Coordinates do not align with features")
        if features.ndim != 2 or features.shape[1] != 2560 or not len(features):
            raise ValueError(f"Invalid feature shape {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite source features")
        return features, mag, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


@torch.inference_mode()
def pool(model, bag):
    hidden = model.projection(bag)
    attention = model.attention(hidden).T.softmax(dim=1)
    pooled = attention @ hidden
    logits = torch.cat(
        [layer(pooled[i : i + 1]) for i, layer in enumerate(model.classifiers)], dim=1
    ).squeeze(0)
    return pooled, logits
