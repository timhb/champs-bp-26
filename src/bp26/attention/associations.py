"""Patient-level cluster associations, uncertainty and distribution summaries."""

from collections import Counter, defaultdict

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score

from bp26.attention.metadata import category_value

DEFAULTS = dict(bootstrap_resamples=200, age_bin_edges=[0, 1, 12, 36, 60])
PROPERTIES = ("age_months", "sex", "age_group", "site")


def patient_rows(labels, nodes, metadata):
    """One dominant cluster per mapped patient; conflicting properties remain unknown."""
    patients = {}
    unmapped = 0
    for label, node, record in zip(labels, nodes, metadata, strict=True):
        patient = record.get("patient_id")
        if not patient:
            unmapped += 1
            continue
        group = patients.setdefault(patient, dict(clusters=Counter(), values=defaultdict(set)))
        group["clusters"][int(label)] += 1
        for key in PROPERTIES:
            value = node.get(key) if key == "site" else record.get(key)
            if key != "age_months":
                value = category_value(value)
            if value is not None and value != "":
                group["values"][key].add(value)
    rows, multiple, ties = [], 0, 0
    for group in patients.values():
        counts = group["clusters"]
        multiple += len(counts) > 1
        winners = [label for label, count in counts.items() if count == max(counts.values())]
        ties += len(winners) > 1
        label = min(winners, key=lambda value: (value < 0, value))
        row = dict(cluster=label)
        for key in PROPERTIES:
            values = group["values"][key]
            row[key] = next(iter(values)) if len(values) == 1 else None
        rows.append(row)
    return rows, dict(
        mapped_patients=len(rows),
        unmapped_slides=unmapped,
        multiple_cluster_patients=multiple,
        tied_patients=ties,
    )


def contingency(clusters, values):
    _, x = np.unique(clusters, return_inverse=True)
    _, y = np.unique(values, return_inverse=True)
    table = np.zeros((int(x.max()) + 1, int(y.max()) + 1)) if len(x) else np.zeros((0, 0))
    np.add.at(table, (x, y), 1)
    return table


def cramers_v(clusters, values):
    table = contingency(clusters, values)
    n = table.sum()
    dimension = min(table.shape, default=0) - 1
    if n == 0 or dimension <= 0:
        return None
    expected = table.sum(axis=1)[:, None] * table.sum(axis=0)[None, :] / n
    chi2 = np.sum((table - expected) ** 2 / expected)
    return float(np.sqrt(np.clip(chi2 / (n * dimension), 0, 1)))


def eta_squared(clusters, values):
    if len(values) < 2 or len(np.unique(clusters)) < 2:
        return None
    mean = np.mean(values)
    total = np.sum((values - mean) ** 2)
    if total <= 0:
        return None
    _, groups = np.unique(clusters, return_inverse=True)
    counts = np.bincount(groups)
    means = np.bincount(groups, weights=values) / counts
    return float(np.clip(np.sum(counts * (means - mean) ** 2) / total, 0, 1))


def site_adjusted(clusters, values, sites, numeric):
    if len(values) < 2 or len(np.unique(clusters)) < 2:
        return None
    if not numeric:
        information = sum(
            np.sum(sites == site)
            / len(sites)
            * mutual_info_score(clusters[sites == site], values[sites == site])
            for site in np.unique(sites)
        )
        return float(max(0, information / np.log(2)))
    # Nested models age ~ site and age ~ site + cluster, without dense one-hot arrays.
    _, s = np.unique(sites, return_inverse=True)
    _, c = np.unique(clusters, return_inverse=True)
    site_counts = np.bincount(s)
    means = np.bincount(s, weights=values) / site_counts
    residual0 = np.sum((values - means[s]) ** 2)
    within_site_variation = any(
        len(np.unique(c[s == site])) > 1 for site in range(len(site_counts))
    )
    if not within_site_variation or residual0 <= np.sum(values**2) * 1e-12:
        return None
    design = sparse.csr_matrix(
        (
            np.ones(2 * len(s)),
            (np.tile(np.arange(len(s)), 2), np.concatenate((s, c + len(site_counts)))),
        ),
        shape=(len(s), len(site_counts) + int(c.max()) + 1),
    )
    coefficients = lsmr(design, values, atol=1e-10, btol=1e-10)[0]
    residual1 = np.sum((values - design @ coefficients) ** 2)
    return float(np.clip((residual0 - residual1) / residual0, 0, 1))


def bootstrap_interval(clusters, values, statistic, resamples, seed):
    if not len(values) or statistic(clusters, values) is None:
        return None
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(resamples):
        indices = rng.integers(0, len(values), len(values))
        value = statistic(clusters[indices], values[indices])
        if value is not None:
            estimates.append(value)
    if len(estimates) < max(20, resamples * 0.8):
        return None
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def property_association(rows, key, settings, seed):
    numeric = key == "age_months"
    known = [row for row in rows if row[key] is not None]
    clusters = np.array([row["cluster"] for row in known], dtype=int)
    values = np.array([row[key] for row in known], dtype=float if numeric else str)
    statistic = eta_squared if numeric else cramers_v
    effect = statistic(clusters, values)
    adjusted_rows = [row for row in known if row["site"] is not None]
    adjusted = None
    if key != "site":
        adjusted = site_adjusted(
            np.array([row["cluster"] for row in adjusted_rows], dtype=int),
            np.array([row[key] for row in adjusted_rows], dtype=float if numeric else str),
            np.array([row["site"] for row in adjusted_rows], dtype=str),
            numeric,
        )
    labels = sorted({row["cluster"] for row in rows})
    if numeric:
        edges = settings["age_bin_edges"]
        categories = [
            f"{low:g}–<{high:g} months" for low, high in zip(edges, edges[1:], strict=False)
        ] + [f"{edges[-1]:g}+ months"]
    else:
        categories = sorted({row[key] for row in known})
    counts = []
    for label in labels:
        group = [row for row in rows if row["cluster"] == label]
        observed = [row[key] for row in group if row[key] is not None]
        if numeric:
            bins = np.histogram(observed, bins=[*settings["age_bin_edges"], np.inf])[0].tolist()
        else:
            frequencies = Counter(observed)
            bins = [frequencies[category] for category in categories]
        counts.append(dict(cluster=label, counts=bins, missing=len(group) - len(observed)))
    all_clusters = np.array([row["cluster"] for row in rows], dtype=int)
    missing = np.array([row[key] is None for row in rows], dtype=int)
    return dict(
        kind="numeric" if numeric else "categorical",
        metric="eta_squared" if numeric else "cramers_v",
        value=effect,
        ci=bootstrap_interval(clusters, values, statistic, settings["bootstrap_resamples"], seed),
        ami=float(adjusted_mutual_info_score(clusters, values))
        if not numeric and effect is not None
        else None,
        site_adjusted=adjusted,
        site_adjusted_n=len(adjusted_rows),
        site_adjusted_metric="partial_eta_squared" if numeric else "conditional_mi_bits",
        known=len(known),
        missing=len(rows) - len(known),
        missingness_v=cramers_v(all_clusters, missing),
        categories=categories,
        distributions=counts,
    )


def cluster_associations(labels, nodes, metadata, settings, seed, site="", include_noise=False):
    rows, coverage = patient_rows(labels, nodes, metadata)
    sites = sorted({row["site"] for row in rows if row["site"] is not None})
    selected = [row for row in rows if not site or row["site"] == site]
    noise = sum(row["cluster"] < 0 for row in selected)
    if not include_noise:
        selected = [row for row in selected if row["cluster"] >= 0]
    return dict(
        cohort=dict(coverage, included=len(selected), noise_patients=noise),
        sites=sites,
        site=site,
        include_noise=include_noise,
        bootstrap_resamples=settings["bootstrap_resamples"],
        seed=seed,
        properties={key: property_association(selected, key, settings, seed) for key in PROPERTIES},
        unit="patient",
        assignment="Most frequent slide cluster per patient; ties choose the smallest assigned cluster, with noise last. Conflicting patient properties remain unknown.",
    )
