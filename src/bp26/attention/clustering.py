"""Branch-specific clustering in the normalized embedding space, never the 2D layout."""

import hashlib
import json
import random
import threading
import uuid
from pathlib import Path

import igraph as ig
import numpy as np
from flask import abort, jsonify, request
from scipy import sparse
from sklearn.cluster import HDBSCAN, KMeans
from threadpoolctl import threadpool_limits

from bp26.attention.associations import DEFAULTS as ASSOCIATION_DEFAULTS
from bp26.attention.associations import cluster_associations
from bp26.attention.metadata import cluster_summaries, load_metadata

LEIDEN_LOCK = threading.Lock()
DEFAULTS = dict(
    default_method="leiden",
    leiden_k=15,
    resolution=1.0,
    max_leiden_edges=2000000,
    min_cluster_size=50,
    min_samples=10,
    n_clusters=12,
    kmeans_n_init=10,
)


def parameters(method, values, n, settings):
    def integer(key, default, minimum, maximum):
        try:
            result = int(str(values.get(key, default)))
        except ValueError:
            raise ValueError(f"{key} must be an integer") from None
        if not minimum <= result <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
        return result

    if method == "leiden":
        k = integer(
            "leiden_k", settings["leiden_k"], 1, min(n - 1, settings["max_leiden_edges"] // n)
        )
        try:
            resolution = float(values.get("resolution", settings["resolution"]))
        except ValueError:
            raise ValueError("Resolution must be a positive finite number") from None
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("Resolution must be a positive finite number")
        return dict(leiden_k=k, resolution=resolution)
    if method == "hdbscan":
        return dict(
            min_cluster_size=integer("min_cluster_size", settings["min_cluster_size"], 2, n),
            min_samples=integer("min_samples", settings["min_samples"], 1, n),
        )
    if method == "kmeans":
        return dict(n_clusters=integer("n_clusters", settings["n_clusters"], 2, n))
    raise ValueError("Choose leiden, hdbscan or kmeans")


def canonical_labels(labels):
    labels = np.asarray(labels)
    groups = sorted(
        (label for label in set(labels) if label >= 0),
        key=lambda label: int(np.flatnonzero(labels == label)[0]),
    )
    mapping = {label: i for i, label in enumerate(groups)}
    return np.array([mapping.get(label, -1) for label in labels], dtype=int)


def leiden(vectors, rankings, k, resolution, seed):
    n = len(vectors)
    targets = np.asarray(rankings[:, :k], dtype=np.int32)
    weights = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, 64):
        stop = min(start + 64, n)
        similarity = np.einsum("ij,ikj->ik", vectors[start:stop], vectors[targets[start:stop]])
        weights[start:stop] = 1 / (1 + np.clip(1 - similarity, 0, 2))
    adjacency = sparse.csr_matrix(
        (weights.ravel(), (np.repeat(np.arange(n), k), targets.ravel())), shape=(n, n)
    )
    upper = sparse.triu(adjacency.maximum(adjacency.T), k=1, format="coo")
    graph = ig.Graph(
        n=n, edges=zip(upper.row.tolist(), upper.col.tolist(), strict=True), directed=False
    )
    with LEIDEN_LOCK:
        ig.set_random_number_generator(random.Random(seed))
        try:
            result = graph.community_leiden(
                objective_function="modularity",
                weights=upper.data.tolist(),
                resolution=resolution,
                n_iterations=-1,
            )
        finally:
            ig.set_random_number_generator(None)
    return result.membership


class ClusterStore:
    def __init__(self, graph):
        self.graph = graph
        self.settings = DEFAULTS | graph.config.get("clustering", {})
        self.metadata_cache = {}
        self.association_settings = ASSOCIATION_DEFAULTS | graph.config.get("associations", {})
        resamples = self.association_settings["bootstrap_resamples"]
        edges = np.asarray(self.association_settings["age_bin_edges"], dtype=float)
        if not isinstance(resamples, int) or resamples < 20:
            raise ValueError("associations.bootstrap_resamples must be an integer of at least 20")
        if (
            edges.ndim != 1
            or len(edges) < 2
            or edges[0] != 0
            or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError("associations.age_bin_edges must start at zero and increase strictly")

    def metadata(self, organ, branch):
        data = self.graph.dataset(organ, branch)
        key = (organ, tuple(node["id"] for node in data["nodes"]))
        if key not in self.metadata_cache:
            self.metadata_cache[key] = load_metadata(self.graph.config, data["nodes"])
        return self.metadata_cache[key]

    def run(self, organ, branch, method, options):
        data = self.graph.dataset(organ, branch)
        identity = dict(
            dataset=data["digest"],
            method=method,
            parameters=options,
            seed=self.graph.settings["seed"],
            n_init=self.settings["kmeans_n_init"],
            version=1,
        )
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        root = Path(self.graph.settings["cache_root"])
        root.mkdir(parents=True, exist_ok=True)
        cache = root / f"clusters-{digest}.json"
        if cache.exists():
            result = json.loads(cache.read_text())
        else:
            vectors = data["vectors"]
            with threadpool_limits(limits=self.graph.config.get("cpu_threads", 4)):
                if method == "leiden":
                    labels = leiden(
                        vectors,
                        self.graph.rankings(organ, branch),
                        options["leiden_k"],
                        options["resolution"],
                        identity["seed"],
                    )
                elif method == "hdbscan":
                    labels = HDBSCAN(
                        **options,
                        metric="euclidean",
                        algorithm="kd_tree",
                        copy=True,
                        n_jobs=self.graph.config.get("cpu_threads", 4),
                    ).fit_predict(vectors)
                else:
                    labels = KMeans(
                        **options,
                        random_state=identity["seed"],
                        n_init=self.settings["kmeans_n_init"],
                    ).fit_predict(vectors)
            labels = canonical_labels(labels).tolist()
            result = dict(
                labels=labels,
                method=method,
                parameters=options,
                seed=identity["seed"],
                n_clusters=len(set(labels) - {-1}),
                noise_count=labels.count(-1),
                representation="512D L2-normalized branch embeddings; Euclidean for K-means/HDBSCAN; symmetrized cosine kNN with similarity 1/(1+distance) for Leiden",
                identity=identity,
            )
            temporary = cache.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temporary.write_text(json.dumps(result))
            temporary.replace(cache)
        metadata = self.metadata(organ, branch)
        return dict(
            result,
            summaries=cluster_summaries(result["labels"], data["nodes"], metadata["records"]),
            metadata_warnings=metadata["warnings"],
        )

    def associations(self, organ, branch, method, options, site, include_noise):
        result = self.run(organ, branch, method, options)
        data = self.graph.dataset(organ, branch)
        settings = self.association_settings
        with threadpool_limits(limits=self.graph.config.get("cpu_threads", 4)):
            analysis = cluster_associations(
                result["labels"],
                data["nodes"],
                self.metadata(organ, branch)["records"],
                settings,
                self.graph.settings["seed"],
                site,
                include_noise,
            )
        return dict(analysis, method=method, parameters=options, provenance=data["provenance"])


def register_clustering(app, graph):
    store = ClusterStore(graph)
    app.extensions["cluster_store"] = store

    def selection():
        organ = request.args.get("organ", graph.settings.get("default_organ", "liver"))
        try:
            graph.source(organ)
            branch = int(request.args.get("branch", "0"))
            data = graph.dataset(organ, branch)
        except ValueError as error:
            abort(400, description=str(error))
        return organ, branch, data

    @app.get("/api/graph/clustering")
    def clustering_options():
        return jsonify(store.settings)

    @app.get("/api/graph/metadata")
    def metadata():
        organ, branch, _ = selection()
        response = jsonify(store.metadata(organ, branch))
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/graph/clusters")
    def clusters():
        organ, branch, data = selection()
        method = request.args.get("method", store.settings["default_method"])
        try:
            options = parameters(method, request.args, len(data["nodes"]), store.settings)
        except ValueError as error:
            abort(400, description=str(error))
        key = ("clusters", organ, branch, method, json.dumps(options, sort_keys=True))
        result, code = graph.job(key, lambda: store.run(organ, branch, method, options))
        response = jsonify(result)
        response.headers["Cache-Control"] = "no-store"
        return response, code

    @app.get("/api/graph/associations")
    def associations():
        organ, branch, data = selection()
        method = request.args.get("method", store.settings["default_method"])
        try:
            options = parameters(method, request.args, len(data["nodes"]), store.settings)
        except ValueError as error:
            abort(400, description=str(error))
        site = request.args.get("site", "")
        include_noise = request.args.get("include_noise", "false")
        if include_noise not in {"true", "false"}:
            abort(400, description="include_noise must be true or false")
        key = (
            "associations",
            organ,
            branch,
            method,
            json.dumps(options, sort_keys=True),
            site,
            include_noise,
        )
        result, code = graph.job(
            key,
            lambda: store.associations(
                organ, branch, method, options, site, include_noise == "true"
            ),
        )
        response = jsonify(result)
        response.headers["Cache-Control"] = "no-store"
        return response, code
