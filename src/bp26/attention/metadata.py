"""Case-level graph overlays; ambiguous or missing identifiers never acquire clinical data."""

from pathlib import Path

import numpy as np
import pandas as pd


def unique_mapping(frame, key, value):
    frame = frame[[key, value]].fillna("").astype(str)
    frame = frame.apply(lambda column: column.str.strip())
    frame = frame.loc[frame[key].ne("") & frame[value].ne("")].drop_duplicates()
    return {
        identity: group[value].iloc[0] if len(group) == 1 else None
        for identity, group in frame.groupby(key)
    }


def age_months(record):
    """Reported years/months/days/hours are components, not alternative total ages."""
    values = []
    for key, factor in (
        ("age_years", 12),
        ("age_months", 1),
        ("age_days", 12 / 365.25),
        ("age_hours", 12 / (365.25 * 24)),
    ):
        value = str(record.get(key, "")).strip()
        if not value:
            continue
        try:
            number = float(value)
        except ValueError:
            return None
        if not np.isfinite(number) or number < 0:
            return None
        values.append(number * factor)
    return float(sum(values)) if values else None


def category_value(value):
    value = str(value).strip() if value is not None else ""
    return value if value and value.casefold() != "unknown" else None


def load_metadata(config, nodes):
    settings = config.get("graph_metadata", {})
    root = Path(config["data_root"])
    warnings = []
    mapping, demographics = {}, {}
    try:
        mapping = unique_mapping(
            pd.read_csv(
                root / settings.get("mapping", "champs_deid_studyid_mapping.csv"),
                dtype=str,
                keep_default_na=False,
            ),
            "study_id",
            "champs_deid",
        )
    except (OSError, ValueError, KeyError) as error:
        warnings.append(f"Clinical case mapping unavailable: {error}")
    try:
        registry = Path(config["slide_registry"])
        if registry.is_file():
            fallback = unique_mapping(
                pd.read_parquet(registry, columns=["study_id", "champs_deid"]),
                "study_id",
                "champs_deid",
            )
            mapping = fallback | mapping
    except (OSError, ValueError, KeyError) as error:
        warnings.append(f"Registry case mapping unavailable: {error}")
    try:
        frame = pd.read_csv(
            root
            / settings.get(
                "demographics",
                "level-2-deidentified-data/CHAMPS_de_identified_data/CHAMPS_deid_basic_demographics.csv",
            ),
            dtype=str,
            keep_default_na=False,
        )
        # Conflicting duplicate patient records remain unknown, rather than choosing a row.
        frame = frame.drop_duplicates()
        frame = frame.loc[
            frame["champs_deid"].str.strip().ne("") & ~frame["champs_deid"].duplicated(keep=False)
        ]
        demographics = frame.set_index("champs_deid").to_dict("index")
    except (OSError, ValueError, KeyError) as error:
        warnings.append(f"Demographics unavailable: {error}")
    records = []
    for node in nodes:
        patient = mapping.get(node["case_id"])
        record = demographics.get(patient, {})
        records.append(
            dict(
                id=node["id"],
                patient_id=patient,
                age_months=age_months(record),
                sex=category_value(record.get("sex")),
                age_group=category_value(record.get("age_group")),
            )
        )
    return dict(
        records=records,
        warnings=warnings,
        age_note="Approximate months at death: reported years × 12 + months + days × 12/365.25 + hours × 12/(365.25×24). Missing components contribute zero only when another component is recorded.",
    )


def cluster_summaries(labels, nodes, metadata):
    summaries = []
    for label in sorted(set(labels)):
        indices = np.flatnonzero(np.asarray(labels) == label)
        patients = {}
        sites = {}
        for i in indices:
            row = metadata[i]
            if row["patient_id"]:
                patients[row["patient_id"]] = row
            site = nodes[i]["site"]
            sites[site] = sites.get(site, 0) + 1
        ages = [row["age_months"] for row in patients.values() if row["age_months"] is not None]
        bins, _ = np.histogram(ages, bins=[0, 1, 12, 36, 60, np.inf])
        summaries.append(
            dict(
                cluster=int(label),
                slides=len(indices),
                patients=len(patients),
                patients_with_age=len(ages),
                patients_missing_age=len(patients) - len(ages),
                unmapped_slides=sum(metadata[i]["patient_id"] is None for i in indices),
                median_age_months=float(np.median(ages)) if ages else None,
                age_q1=float(np.quantile(ages, 0.25)) if ages else None,
                age_q3=float(np.quantile(ages, 0.75)) if ages else None,
                age_bins=bins.tolist(),
                sites=sites,
            )
        )
    return summaries
