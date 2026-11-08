"""Exact cosine neighborhoods of completed, branch-specific slide embeddings."""

import hashlib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from flask import abort, jsonify, request
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from bp26.inference.champion import load_model, pool, read_bag


class GraphStore:
    def __init__(self, config, catalog):
        self.config, self.catalog = config, catalog
        self.settings = config.get("graph", {})
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="slide-graph")
        self.jobs = {}

    def source(self, organ):
        if organ not in self.settings.get("organs", ["liver"]):
            raise ValueError("Organ is not configured for the graph")
        if "embeddings_root" not in self.settings:
            raise ValueError("Configure graph.embeddings_root to a completed CLAM export")
        return Path(self.settings["embeddings_root"]) / f"{organ}.h5"

    def overview(self):
        organs = []
        for organ in self.settings.get("organs", ["liver"]):
            path = self.source(organ)
            with h5py.File(path, "r") as handle:
                ids = handle["slide_ids"].asstr()[:]
                completed = handle["completed"][:].astype(bool)
                known = set(self.catalog.loc[self.catalog.organ.eq(organ), "slide_id"])
                count = sum(done and sid in known for sid, done in zip(ids, completed, strict=True))
                organs.append(
                    dict(
                        organ=organ,
                        count=int(sum(self.catalog.organ.eq(organ)))
                        if self.settings.get("recover_missing")
                        else int(count),
                        exported_count=int(count),
                        branches=handle["branch_labels"].asstr()[:].tolist(),
                    )
                )
        return dict(
            organs=organs,
            default_organ=self.settings.get("default_organ", "liver"),
            default_k=self.settings.get("default_k", 10),
        )

    def job(self, key, operation):
        """Bound compute concurrency; failed jobs can be retried explicitly."""
        with self.lock:
            if key not in self.jobs:
                # Keep only the current edge preview per organ/branch; projections stay cached.
                if key[0] == "edges":
                    for old in list(self.jobs):
                        if old[:3] == key[:3] and self.jobs[old].done():
                            del self.jobs[old]
                self.jobs[key] = self.executor.submit(operation)
            future = self.jobs[key]
        if not future.done():
            return dict(status="computing"), 202
        try:
            return dict(status="ready", **future.result()), 200
        except Exception:
            with self.lock:
                self.jobs.pop(key, None)
            raise

    def dataset(self, organ, branch):
        key = ("data", organ, branch)
        with self.lock:
            if key in self.jobs:
                return self.jobs[key].result()
        path = self.source(organ)
        with h5py.File(path, "r") as handle:
            labels = handle["branch_labels"].asstr()[:].tolist()
            if branch < 0 or branch >= len(labels):
                raise ValueError("Finding branch is out of range")
            if handle.attrs.get("organ") != organ or handle.attrs.get("encoder") != "virchow2":
                raise ValueError("Embedding export has incompatible organ/encoder provenance")
            ids = handle["slide_ids"].asstr()[:]
            if len(set(ids)) != len(ids):
                raise ValueError("Embedding export contains duplicate slide identities")
            rows = self.catalog.loc[self.catalog.organ.eq(organ)].set_index("slide_id", drop=False)
            completed = handle["completed"][:].astype(bool)
            positions = [i for i, sid in enumerate(ids) if completed[i] and sid in rows.index]
            vectors = handle["embeddings"][:, branch, :][positions].astype(np.float32)
            mapped_ids = ids[positions].tolist()
            selected = rows.loc[mapped_ids]
            provenance = dict(
                checkpoint_sha256=str(handle.attrs["checkpoint_sha256"]),
                manifest_sha256=str(handle.attrs["manifest_sha256"]),
                branch=labels[branch],
                organ=organ,
            )
        missing = rows.loc[~rows.index.isin(ids)]
        if self.settings.get("recover_missing") and not missing.empty:
            extra = self.recover(organ, missing, labels, provenance["checkpoint_sha256"])
            vectors = np.concatenate([vectors, extra[:, branch, :]])
            mapped_ids.extend(missing.slide_id.tolist())
            selected = rows.loc[mapped_ids]
            provenance["recovered_slides"] = missing.slide_id.tolist()
            provenance["recovered_sha256"] = hashlib.sha256(extra.tobytes()).hexdigest()
        if len(vectors) < 3:
            raise ValueError("At least three mapped, completed slides are required")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(vectors).all() or np.any(norms == 0):
            raise ValueError("Completed embeddings contain nonfinite or zero vectors")
        vectors /= norms[:, None]
        nodes = selected[["id", "slide_id", "case_id", "site", "organ"]].to_dict("records")
        identity = dict(
            provenance,
            ids=mapped_ids,
            source_size=path.stat().st_size,
            source_mtime=path.stat().st_mtime_ns,
            settings=self.settings,
            version=1,
        )
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        result = dict(
            vectors=vectors,
            nodes=nodes,
            provenance=provenance,
            digest=digest,
            catalog_count=len(rows),
            excluded=len(rows) - len(nodes),
        )
        # A resolved Future makes reads safe without a second data cache.
        from concurrent.futures import Future

        future = Future()
        future.set_result(result)
        with self.lock:
            self.jobs[key] = future
        return result

    def recover(self, organ, rows, labels, checkpoint_hash):
        """Fill export gaps from indexed bags using exactly the export's checkpoint."""
        import torch

        sources = [
            dict(
                slide_id=row.slide_id,
                path=row.path,
                size=Path(row.path).stat().st_size,
                mtime=Path(row.path).stat().st_mtime_ns,
            )
            for row in rows.itertuples()
        ]
        identity = dict(checkpoint_sha256=checkpoint_hash, sources=sources, labels=labels)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        root = Path(self.settings["cache_root"])
        root.mkdir(parents=True, exist_ok=True)
        cache = root / f"{organ}-recovered-{digest}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                return saved["vectors"]
        model_root = (
            Path(self.config["models_root"]) / f"focus17_{organ}_virchow2_clam_mb_multilabel"
        )
        directories = list(model_root.glob("*/job_*"))
        if len(directories) != 1:
            raise ValueError("Cannot identify the export checkpoint for missing slides")
        model, current_labels, record = load_model(directories[0], "cpu")
        if record["sha256"] != checkpoint_hash or current_labels != labels:
            raise ValueError("Recovery checkpoint or finding order differs from the export")
        recovered = []
        with threadpool_limits(limits=self.config.get("cpu_threads", 4)):
            for row in rows.itertuples():
                with h5py.File(row.path, "r") as handle:
                    if handle["features"].attrs.get("encoder") != "virchow2":
                        raise ValueError("Recovery requires Virchow2 features")
                features, magnification, error = read_bag(row.path)
                if error or magnification != 20:
                    raise ValueError(
                        f"Cannot recover {row.slide_id}: {error or 'expected 20x features'}"
                    )
                pooled, _ = pool(model, torch.from_numpy(features))
                recovered.append(pooled.numpy())
        vectors = np.stack(recovered)
        temporary = cache.with_suffix(f".{uuid.uuid4().hex}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, vectors=vectors)
        temporary.replace(cache)
        cache.with_suffix(".json").write_text(json.dumps(identity, indent=2) + "\n")
        return vectors

    def projection(self, organ, branch, projection):
        data = self.dataset(organ, branch)
        root = Path(self.settings["cache_root"])
        root.mkdir(parents=True, exist_ok=True)
        cache = root / f"{data['digest']}-{projection}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                coords, variance = saved["coords"], saved["variance"]
        else:
            with threadpool_limits(limits=self.config.get("cpu_threads", 4)):
                vectors = data["vectors"]
                pca = PCA(n_components=min(50, *vectors.shape), random_state=self.settings["seed"])
                reduced = pca.fit_transform(vectors)
                variance = pca.explained_variance_ratio_[:2]
                coords = reduced[:, :2]
                if projection == "tsne":
                    coords = TSNE(
                        n_components=2,
                        perplexity=min(self.settings["perplexity"], len(vectors) - 1),
                        max_iter=self.settings["tsne_iterations"],
                        random_state=self.settings["seed"],
                        init="pca",
                        learning_rate="auto",
                        n_jobs=self.config.get("cpu_threads", 4),
                    ).fit_transform(reduced)
                elif projection == "umap":
                    from umap import UMAP

                    coords = UMAP(
                        n_components=2,
                        n_neighbors=min(self.settings.get("umap_neighbors", 15), len(vectors) - 1),
                        min_dist=self.settings.get("umap_min_dist", 0.1),
                        metric="cosine",
                        init="pca",
                        random_state=self.settings["seed"],
                        n_jobs=1,
                    ).fit_transform(vectors)
            temporary = cache.with_suffix(f".{uuid.uuid4().hex}.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, coords=coords, variance=variance)
            temporary.replace(cache)
        return dict(
            nodes=data["nodes"],
            coords=coords.tolist(),
            variance=variance.tolist(),
            provenance=data["provenance"],
            excluded=data["excluded"],
            catalog_count=data["catalog_count"],
            projection=projection,
            metric="cosine",
            umap_neighbors=min(self.settings.get("umap_neighbors", 15), len(data["nodes"]) - 1),
            umap_min_dist=self.settings.get("umap_min_dist", 0.1),
            seed=self.settings["seed"],
            perplexity=min(self.settings["perplexity"], len(data["nodes"]) - 1),
        )

    @staticmethod
    def validate_k(k, size):
        if k < 1 or k >= size:
            raise ValueError(f"k must be between 1 and {size - 1}")

    def neighbors(self, organ, branch, key, k):
        data = self.dataset(organ, branch)
        self.validate_k(k, len(data["nodes"]))
        lookup = {node["id"]: i for i, node in enumerate(data["nodes"])}
        if key not in lookup:
            raise ValueError("Slide is not in this organ's completed embedding export")
        index = lookup[key]
        with threadpool_limits(limits=self.config.get("cpu_threads", 4)):
            distances = np.clip(1 - data["vectors"] @ data["vectors"][index], 0, 2)
        distances[index] = np.inf
        order = np.argsort(distances, kind="stable")[:k]
        return dict(source=index, indices=order.tolist(), distances=distances[order].tolist(), k=k)

    def edges(self, organ, branch, k):
        data = self.dataset(organ, branch)
        n = len(data["nodes"])
        self.validate_k(k, n)
        per_node = max(1, self.settings["edge_budget"] // n)
        ranks = np.unique(np.linspace(0, k - 1, min(k, per_node), dtype=int))
        order = self.rankings(organ, branch)
        edges = [[i, int(j)] for i, row in enumerate(order[:, ranks]) for j in row]
        return dict(
            edges=edges, k=k, total_edges=n * k, displayed_edges=len(edges), sampled=len(ranks) < k
        )

    def rankings(self, organ, branch):
        data = self.dataset(organ, branch)
        vectors = data["vectors"]
        n = len(vectors)
        root = Path(self.settings["cache_root"])
        root.mkdir(parents=True, exist_ok=True)
        cache = root / f"{data['digest']}-neighbors.npy"
        if not cache.exists():
            temporary = cache.with_suffix(f".{uuid.uuid4().hex}.tmp")
            order = np.lib.format.open_memmap(
                temporary, mode="w+", dtype="uint16" if n <= 65536 else "uint32", shape=(n, n - 1)
            )
            with threadpool_limits(limits=self.config.get("cpu_threads", 4)):
                # Distance blocks stay in RAM; exact ranks live in a compact disk cache.
                for start in range(0, n, 64):
                    distances = np.clip(1 - vectors[start : start + 64] @ vectors.T, 0, 2)
                    distances[
                        np.arange(len(distances)), np.arange(start, start + len(distances))
                    ] = np.inf
                    order[start : start + len(distances)] = np.argsort(
                        distances, axis=1, kind="stable"
                    )[:, :-1]
            order.flush()
            del order
            temporary.replace(cache)
        return np.load(cache, mmap_mode="r", allow_pickle=False)


def register_graph(app, config, catalog):
    store = GraphStore(config, catalog)
    app.extensions["graph_store"] = store
    from bp26.attention.clustering import register_clustering

    register_clustering(app, store)

    def selection():
        organ = request.args.get("organ", store.settings.get("default_organ", "liver"))
        store.source(organ)
        try:
            branch = int(request.args.get("branch", "0"))
        except ValueError:
            abort(400, description="Branch must be an integer")
        with h5py.File(store.source(organ), "r") as handle:
            if branch < 0 or branch >= len(handle["branch_labels"]):
                abort(400, description="Finding branch is out of range")
        return organ, branch

    def k_value():
        try:
            return int(request.args.get("k", "10"))
        except ValueError:
            abort(400, description="k must be an integer")

    @app.get("/api/graph/catalog")
    def graph_catalog():
        return jsonify(store.overview())

    @app.get("/api/graph")
    def graph_projection():
        organ, branch = selection()
        projection = request.args.get("projection", "pca")
        if projection not in {"pca", "tsne", "umap"}:
            abort(400, description="Choose pca, tsne or umap")
        result, code = store.job(
            ("projection", organ, branch, projection),
            lambda: store.projection(organ, branch, projection),
        )
        return jsonify(result), code

    @app.get("/api/graph/edges")
    def graph_edges():
        organ, branch = selection()
        k = k_value()
        data = store.dataset(organ, branch)
        try:
            store.validate_k(k, len(data["nodes"]))
        except ValueError as error:
            abort(400, description=str(error))
        result, code = store.job(("edges", organ, branch, k), lambda: store.edges(organ, branch, k))
        return jsonify(result), code

    @app.get("/api/graph/neighbors/<key>")
    def graph_neighbors(key):
        organ, branch = selection()
        try:
            return jsonify(store.neighbors(organ, branch, key, k_value()))
        except ValueError as error:
            abort(400, description=str(error))
