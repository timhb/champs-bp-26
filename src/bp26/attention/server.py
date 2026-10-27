"""Local HTML viewer for CLAM-MB branch attention. Run `pixi run attention-viewer`."""

import argparse
import csv
import io
import threading
import tomllib
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import openslide
import torch
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from openslide.deepzoom import DeepZoomGenerator

from bp26.attention.data import AttentionStore, load_catalog, percentiles
from bp26.attention.graph import register_graph
from bp26.attention.patient import PatientStore
from bp26.attention.references import ReferenceFindings

REPO = Path(__file__).resolve().parents[3]
UI = REPO / "notebooks/attention-viewer"


def load_config(path):
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    for key in ("data_root", "manifest", "feature_index", "slide_registry", "models_root"):
        value = Path(config[key])
        config[key] = value if value.is_absolute() else REPO / value
    for key in ("attention_cache_size", "open_slide_cache_size", "cpu_threads"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if "graph" in config:
        for key in ("embeddings_root", "cache_root"):
            value = Path(config["graph"][key])
            config["graph"][key] = str(value if value.is_absolute() else REPO / value)
        for key in ("default_k", "edge_budget", "perplexity"):
            if config["graph"][key] < 1:
                raise ValueError(f"graph.{key} must be positive")
        neighbors = config["graph"].get("umap_neighbors", 15)
        if not isinstance(neighbors, int) or neighbors < 2:
            raise ValueError("graph.umap_neighbors must be an integer of at least 2")
        if not 0 <= config["graph"].get("umap_min_dist", 0.1) <= 1:
            raise ValueError("graph.umap_min_dist must be between 0 and 1")
        if config["graph"]["tsne_iterations"] < 250:
            raise ValueError("graph.tsne_iterations must be at least 250")
    return config


class SlideCache:
    def __init__(self, config, catalog):
        self.config, self.catalog = config, catalog
        self.lock, self.entries = threading.RLock(), OrderedDict()

    def get(self, key):
        """Caller holds the lock while using a slide, so eviction cannot close it mid-read."""
        if key not in self.entries:
            path = Path(self.catalog.loc[key, "wsi_path"]).resolve()
            if not path.is_relative_to(Path(self.config["data_root"]).resolve()):
                raise ValueError("WSI is outside the configured data root")
            slide = openslide.OpenSlide(str(path))
            # Full level-0 image, without tissue-bound cropping: matches HDF5 coords.
            self.entries[key] = (
                slide,
                DeepZoomGenerator(slide, tile_size=254, overlap=1, limit_bounds=False),
            )
        self.entries.move_to_end(key)
        while len(self.entries) > self.config["open_slide_cache_size"]:
            _, (old, _) = self.entries.popitem(last=False)
            old.close()
        return self.entries[key]


def image_response(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=88)
    buffer.seek(0)
    response = send_file(buffer, mimetype="image/jpeg", max_age=3600)
    response.headers["Cache-Control"] = "private, max-age=3600"
    return response


def create_app(config, catalog=None):
    app = Flask(__name__, static_folder=None)
    catalog = load_catalog(config) if catalog is None else catalog
    slides, attention = SlideCache(config, catalog), AttentionStore(config, catalog)
    app.extensions["attention_store"] = attention
    references = ReferenceFindings(config)
    app.extensions["reference_findings"] = references
    patients = PatientStore(config, catalog)
    app.extensions["patient_store"] = patients
    register_graph(app, config, catalog)

    def row_for(key):
        if key not in catalog.index:
            abort(404, description="Slide not found in the configured manifest")
        return catalog.loc[key]

    def result_for(key):
        row_for(key)
        with slides.lock:
            dimensions = slides.get(key)[0].dimensions
        return attention.get(key, dimensions)

    @app.errorhandler(Exception)
    def error_response(error):
        from werkzeug.exceptions import HTTPException

        code = error.code if isinstance(error, HTTPException) else 422
        if not isinstance(error, HTTPException):
            app.logger.exception("Viewer request failed")
        message = error.description if isinstance(error, HTTPException) else str(error)
        return jsonify(error=message), code

    @app.get("/")
    def index():
        return send_from_directory(UI, "index.html")

    @app.get("/graph")
    def graph():
        return send_from_directory(UI, "graph.html")

    @app.get("/assets/<path:name>")
    def assets(name):
        return send_from_directory(UI / "assets", name)

    @app.get("/vendor/<path:name>")
    def vendor(name):
        return send_from_directory(UI / "vendor", name)

    @app.get("/api/slides")
    def search():
        frame = catalog
        query = request.args.get("q", "").strip()
        organ, site = request.args.get("organ"), request.args.get("site")
        if query:
            mask = frame.slide_id.str.contains(query, case=False, regex=False)
            mask |= frame.case_id.str.contains(query, case=False, regex=False)
            frame = frame.loc[mask]
        if organ:
            frame = frame.loc[frame.organ.eq(organ)]
        if site:
            frame = frame.loc[frame.site.eq(site)]
        condition_index = references.condition_index(catalog, config.get("models_root"))
        counts = Counter(label for key in frame.index for label in condition_index[key])
        conditions = [dict(value=label, count=count) for label, count in sorted(counts.items())]
        condition = request.args.get("condition", "")
        if condition:
            frame = frame.loc[[condition in condition_index[key] for key in frame.index]]
        try:
            offset = max(0, int(request.args.get("offset", 0)))
        except ValueError:
            abort(400, description="Invalid page offset")
        return jsonify(
            total=len(frame),
            total_available=len(catalog),
            conditions=conditions,
            offset=offset,
            sites=sorted(catalog.site.unique().tolist()),
            slides=frame.iloc[offset : offset + 40][
                ["id", "slide_id", "organ", "site", "case_id"]
            ].to_dict("records"),
        )

    @app.get("/api/slides/<key>")
    def detail(key):
        row = row_for(key)
        result = result_for(key)
        with slides.lock:
            slide = slides.get(key)[0]
            properties = slide.properties

        def positive_float(value):
            try:
                value = float(value)
                return value if np.isfinite(value) and value > 0 else None
            except ValueError, TypeError:
                return None

        reference = references.get(row, result["labels"], result["checkpoint"]["checkpoint"])
        return jsonify(
            id=key,
            slide_id=row.slide_id,
            organ=row.organ,
            site=row.site,
            case_id=row.case_id,
            dimensions=list(result["dimensions"]),
            patch_size=result["patch_size"],
            n_tiles=len(result["coords"]),
            encoder="Virchow2",
            magnification=20,
            objective=positive_float(properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)),
            mpp_x=positive_float(properties.get(openslide.PROPERTY_NAME_MPP_X)),
            reference=reference,
            checkpoint=Path(result["checkpoint"]["checkpoint"]).name,
            epoch=result["checkpoint"]["epoch"],
            branches=[
                dict(
                    id=i,
                    label=label,
                    score=float(result["scores"][i]),
                    logit=float(result["logits"][i]),
                    reference=reference["states"][label],
                )
                for i, label in enumerate(result["labels"])
            ],
        )

    @app.get("/api/slides/<key>/case")
    def patient_case(key):
        row = row_for(key)
        response = jsonify(patients.get(row))
        response.headers["Cache-Control"] = "no-store"
        return response

    def branch_result(key, branch):
        result = result_for(key)
        if branch < 0 or branch >= len(result["labels"]):
            abort(404, description="Finding branch not found")
        weights = result["attention"][branch]
        return result, weights, percentiles(weights)

    @app.get("/api/slides/<key>/attention/<int:branch>")
    def heatmap(key, branch):
        result, weights, ranks = branch_result(key, branch)
        top = np.argsort(-weights, kind="stable")[:8]
        return jsonify(
            branch=branch,
            label=result["labels"][branch],
            coords=result["coords"].tolist(),
            weights=weights.tolist(),
            percentiles=ranks.tolist(),
            patch_size=result["patch_size"],
            top_tiles=top.tolist(),
        )

    @app.get("/api/slides/<key>/export/<int:branch>.csv")
    def export(key, branch):
        result, weights, ranks = branch_result(key, branch)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "slide_id",
                "finding",
                "tile_index",
                "x_level0",
                "y_level0",
                "patch_size_level0",
                "attention",
                "percentile",
            ]
        )
        for i, (coords, weight, rank) in enumerate(
            zip(result["coords"], weights, ranks, strict=True)
        ):
            writer.writerow(
                [
                    catalog.loc[key, "slide_id"],
                    result["labels"][branch],
                    i,
                    *coords,
                    result["patch_size"],
                    float(weight),
                    float(rank) * 100,
                ]
            )
        response = app.response_class(buffer.getvalue(), mimetype="text/csv")
        response.headers["Content-Disposition"] = (
            f'attachment; filename="{key}-branch-{branch}.csv"'
        )
        return response

    @app.get("/api/slides/<key>/patch/<int:tile>.jpg")
    def patch(key, tile):
        result = result_for(key)
        if tile < 0 or tile >= len(result["coords"]):
            abort(404, description="Tile not found")
        with slides.lock:
            slide = slides.get(key)[0]
            size = result["patch_size"]
            level = slide.get_best_level_for_downsample(size / 224)
            edge = max(1, round(size / slide.level_downsamples[level]))
            coords = tuple(map(int, result["coords"][tile]))
            image = slide.read_region(coords, level, (edge, edge)).convert("RGB")
        image.thumbnail((224, 224))
        return image_response(image)

    @app.get("/dzi/<key>.dzi")
    def dzi(key):
        row_for(key)
        with slides.lock:
            xml = slides.get(key)[1].get_dzi("jpeg")
        return app.response_class(xml, mimetype="application/xml")

    @app.get("/dzi/<key>_files/<int:level>/<int:col>_<int:row>.jpeg")
    def tile(key, level, col, row):
        row_for(key)
        with slides.lock:
            try:
                image = slides.get(key)[1].get_tile(level, (col, row))
            except ValueError:
                abort(404, description="Tile not found")
        return image_response(image)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/attention-viewer.toml")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(config["cpu_threads"])
    app = create_app(config)
    print(f"Attention viewer: http://{config['host']}:{args.port or config['port']}", flush=True)
    app.run(host=config["host"], port=args.port or config["port"], threaded=True, debug=False)
