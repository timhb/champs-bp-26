"""Inference defaults kept in configs/inference/champs.toml."""

import tomllib
from pathlib import Path


def settings(path=None):
    path = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[3] / "configs/inference/champs.toml"
    )
    with path.open("rb") as stream:
        return tomllib.load(stream)
