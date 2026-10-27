"""Links each slide to the CHAMPS case-level tables (read-only).

slide folder  BDAA00125                      (study_id)
   -> champs_deid_studyid_mapping.csv        (study_id -> champs_deid)
      fallback: reports/derived/slide_labels.parquet (also carries both ids)
   -> Level-2 tables + CPL reports, all keyed by champs_deid
slide file stem -> slide_labels.parquet slide_id  (organ / semantic labels)
"""

import os
import re
import threading

import pandas as pd

if __package__:
    from .pathology import CONDITIONS, slide_finding_index
    from .reports import REPORT_FIELDS, format_mits_report
    from .reports import clean_report_text as _clean_text
else:
    from pathology import CONDITIONS, slide_finding_index
    from reports import REPORT_FIELDS, format_mits_report
    from reports import clean_report_text as _clean_text

PATHS = {}  # set by configure(): root, l2 (Level-2 tables), reports
state = {"ready": False, "error": None}
D = {}  # name -> dict[champs_deid] -> list[dict] / dict

LAB_SPECIMEN = {"bld": "Blood", "csf": "CSF"}
TAC_SPECIMEN = {
    "bld": "Blood",
    "bld_sp": "Blood (spun)",
    "CSF": "CSF",
    "lung": "Lung",
    "rect_swab": "Rectal swab",
    "NP_OP": "NP/OP swab",
}
CODE_RE = re.compile(r"\bCH\d{5}\b", re.I)
TAC_RE = re.compile(r"^(?P<spec>.+)_(?P<code>[A-Z0-9]{4}_\d+)$")
SLIDE_CODE_RE = re.compile(r"M\d{5}[._](\d{3})")  # tissue code, e.g. M01210.044 -> 044
ICD_COLS_DESC = {
    "Immediate_COD": "IC_champs_group_desc",
    "Underlying_Cause": "UC_champs_group_desc",
}
LEVEL2_TABLES = [
    "basic_demographics",
    "decode_results",
    "lab_results",
    "tac_results",
    "verbal_autopsy",
]
CPL_PREFIXES = {
    "Diagnosis": "deid_CPL_DB_Diagnosis",
    "IHC": "deid_CPL_DB_IHC",
    "SpecialStains": "deid_CPL_DB_SpecialStains",
    "PCR": "deid_CPL_DB_PCR",
}
ADEQUACY_COLS = [
    "section",
    "organ",
    "subsite",
    "present_count",
    "total_count",
    "adequate",
    "technically_inadequate",
    "no_target_tissue",
    "match_text",
]
SLIDE_LABEL_COLS = ["study_id", "slide_id", "source", "organ", "tissue_code", "semantic_groups"]


def configure(data_root):
    PATHS["root"] = data_root
    PATHS["l2"] = f"{data_root}/level-2-deidentified-data/CHAMPS_de_identified_data"
    PATHS["reports"] = f"{data_root}/reports"


# ---------------------------------------------------------------- loading
def _read(path, **kw):
    return pd.read_csv(
        path, dtype=str, keep_default_na=False, na_values=[""], low_memory=False, **kw
    )


def _by_deid(df, cols=None):
    """Group rows by champs_deid; tables with several rows per case keep them as a list."""
    out = {}
    for rec in df.to_dict("records"):
        rec = {k: (v if isinstance(v, str) else None) for k, v in rec.items()}
        key = rec.pop("champs_deid")
        if not key or not key.strip():
            continue
        if cols:
            rec = {c: rec.get(c) for c in cols}
        out.setdefault(key, []).append(rec)
    return out


def _tac_dictionary():
    """Assay code -> name, parsed from the dataset description PDF."""
    try:
        import pypdf

        pdf = next(f for f in os.listdir(PATHS["l2"]) if f.lower().endswith(".pdf"))
        reader = pypdf.PdfReader(os.path.join(PATHS["l2"], pdf))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:  # names are cosmetic; the viewer still shows the raw codes
        return {}

    names = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Z0-9]{4}_\d)\s+(.+?)\s*$", line)
        if m:
            names.setdefault(m.group(1), re.sub(r"\s{2,}", " · ", m.group(2)))
    return names


def _load_lookups():
    voc = _read(f"{PATHS['l2']}/CHAMPS_vocabulary.csv")
    D["vocab"] = dict(zip(voc.champs_local_code.str.upper(), voc.c_name, strict=True))

    icd = _read(f"{PATHS['l2']}/CHAMPS_icd_descriptions.csv")
    D["icd"] = dict(zip(icd.ICD_code.str.strip(), icd.ICD_desc, strict=True))

    D["tac_names"] = _tac_dictionary()


def _load_ids_and_slide_labels():
    """study_id -> champs_deid from the mapping CSV, filled in from slide_labels where missing."""
    m = _read(f"{PATHS['root']}/champs_deid_studyid_mapping.csv")
    study2deid = {
        sid: did
        for sid, did in zip(m.study_id, m.champs_deid, strict=True)
        if isinstance(sid, str) and sid.strip() and isinstance(did, str) and did.strip()
    }

    sl = pd.read_parquet(f"{PATHS['reports']}/derived/slide_labels.parquet")
    pairs = sl[["study_id", "champs_deid"]].drop_duplicates().itertuples(index=False)
    for sid, did in pairs:
        if (
            isinstance(sid, str)
            and sid.strip()
            and isinstance(did, str)
            and did.strip()
            and sid not in study2deid
        ):
            study2deid[sid] = did
    D["study2deid"] = study2deid

    sl = sl.where(pd.notna(sl), None)
    D["slide_findings"] = slide_finding_index(
        sl[["wsi_path", "organ", "semantic_groups"]].to_dict("records"), PATHS["root"]
    )
    records = sl[SLIDE_LABEL_COLS].to_dict("records")
    D["slide_labels"] = {(r["study_id"], r["slide_id"]): r for r in records}

    # tissue code (e.g. 044) -> organ, majority vote from the labelled slides, ignoring "other"
    labelled = sl[sl.organ != "other"]
    tc = labelled.groupby("tissue_code").organ.agg(lambda x: x.value_counts().index[0])
    D["tissue_organ"] = {str(k).zfill(3): v for k, v in tc.items()}


def _load_level2_tables():
    """One row per case; empty fields are dropped so only recorded values are shown."""
    for name in LEVEL2_TABLES:
        df = _read(f"{PATHS['l2']}/CHAMPS_deid_{name}.csv")
        D[name] = {
            r.pop("champs_deid"): {k: v for k, v in r.items() if isinstance(v, str) and v.strip()}
            for r in df.to_dict("records")
        }


def _load_cpl_reports():
    """Newest export per CPL report type (file names carry a timestamp)."""
    reports = PATHS["reports"]
    for key, prefix in CPL_PREFIXES.items():
        candidates = [x for x in os.listdir(reports) if x.startswith(prefix) and x.endswith(".csv")]
        newest = sorted(candidates)[-1]
        D["cpl_" + key] = _by_deid(_read(f"{reports}/{newest}"))
        D.setdefault("sources", {})["cpl_" + key] = newest


def _load_derived():
    adequacy = _read(f"{PATHS['reports']}/derived/adequacy_statements.csv")
    D["adequacy"] = _by_deid(adequacy, ADEQUACY_COLS)

    pigment_file = "derived/pigment_confirmation/pigment_confirmation.parquet"
    pg = pd.read_parquet(f"{PATHS['root']}/{pigment_file}")
    D["pigment"] = {r.pop("champs_deid"): r for r in pg.astype(str).to_dict("records")}


def load():
    try:
        _load_lookups()
        _load_ids_and_slide_labels()
        _load_level2_tables()
        _load_cpl_reports()
        _load_derived()
        state["ready"] = True
    except Exception as e:  # keep the image viewer working even if tables fail
        state["error"] = repr(e)
        raise


def start():
    threading.Thread(target=load, daemon=True).start()


# ---------------------------------------------------------------- per slide
def _stem(filename):
    return re.sub(r"\.svs$", "", filename, flags=re.I)


def slide_meta(case, filename):
    """organ / CPL labels for one slide + whether case-level data exists."""
    if not state["ready"]:
        return {}

    stem = _stem(filename)
    label = D["slide_labels"].get((case, stem)) or {}
    code = SLIDE_CODE_RE.search(stem)
    organ_from_code = D["tissue_organ"].get(code.group(1)) if code else None
    organ = label.get("organ") or organ_from_code or "other"

    deid = D["study2deid"].get(case)
    tac = D["tac_results"].get(deid, {})
    return {
        "organ": organ,
        "deid": deid,
        "has_data": bool(deid and deid in D["basic_demographics"]),
        "tac_pos": bool(deid and any(v == "Positive" for v in tac.values())),
    }


def _decode_codes(s):
    return CODE_RE.sub(lambda m: D["vocab"].get(m.group(0).upper(), m.group(0)), s)


def _lab(rec):
    """Lab columns look like bld_CH00603_CH00158: specimen, then vocabulary codes."""
    rows = []
    for col, val in rec.items():
        spec, _, rest = col.partition("_CH") if "_CH" in col else col.partition("_ch")
        spec = LAB_SPECIMEN.get(spec, spec.replace("_", " "))
        test = _decode_codes(("CH" + rest) if rest else col).replace("_", " · ")
        main, _, detail = _decode_codes(val).partition(":")
        rows.append(
            {
                "specimen": spec,
                "test": test,
                "result": main.strip(),
                "detail": detail.strip(),
                "col": col,
            }
        )
    return rows


def _tac(rec):
    """TAC columns look like bld_KLPN_1: specimen, assay code."""
    rows = []
    for col, val in rec.items():
        m = TAC_RE.match(col)
        spec, code = (m.group("spec"), m.group("code")) if m else ("", col)
        target = D["tac_names"].get(code, "")
        rows.append(
            {
                "specimen": TAC_SPECIMEN.get(spec, spec),
                "code": code,
                "target": target,
                "result": val,
            }
        )
    return rows


def _group_col(col):
    default = col.replace("Morbid_Condition_", "Morbid_Cond_") + "_champs_group_desc"
    return ICD_COLS_DESC.get(col) or default


def _decode(rec):
    out = []
    for col, val in rec.items():
        if col.endswith("_desc"):
            continue
        desc = D["icd"].get(val.strip(), "")
        out.append(
            {
                "field": col.replace("_", " "),
                "value": val,
                "desc": desc,
                "group": rec.get(_group_col(col)),
            }
        )
    return out


def _dedupe_report(rows):
    """CPL tables repeat the full diagnosis on every row; drop it and keep unique result rows."""
    skip = {"Specimen Info", "Diagnosis", "DiagnosisComment", "HEText", "champs_deid"}
    items = [
        {
            k: (_clean_text(v) if isinstance(v, str) else v)
            for k, v in r.items()
            if k not in skip and v
        }
        for r in rows
    ]

    seen, uniq = set(), []
    for item in items:
        key = tuple(sorted(item.items()))
        if item and key not in seen:
            seen.add(key)
            uniq.append(item)
    return uniq


def _case_tables(deid):
    dec = D["decode_results"].get(deid)
    lab = D["lab_results"].get(deid)
    tac = D["tac_results"].get(deid)
    diag = D["cpl_Diagnosis"].get(deid, [])
    out = {
        "demographics": D["basic_demographics"].get(deid),
        "decode": _decode(dec) if dec else None,
        "lab": _lab(lab) if lab else None,
        "tac": _tac(tac) if tac else None,
        "verbal_autopsy": D["verbal_autopsy"].get(deid),
        "cpl_diagnosis": [{k: _clean_text(v) for k, v in r.items() if v} for r in diag],
        "adequacy": D["adequacy"].get(deid),
        "pigment": D["pigment"].get(deid),
        "sources": D.get("sources"),
        "mits_reports": _mits_reports(deid),
    }

    for key in ("IHC", "SpecialStains", "PCR"):
        out["cpl_" + key] = _dedupe_report(D["cpl_" + key].get(deid, [])) or None
    return out


def _mits_reports(deid):
    """Keep narrative variants from every export, merging only identical shared text."""
    reports = {}
    for key in CPL_PREFIXES:
        source = D.get("sources", {}).get("cpl_" + key, key)
        for row in D["cpl_" + key].get(deid, []):
            identity = tuple(_clean_text(row.get(field)) for field in REPORT_FIELDS)
            if not any(identity[1:]):
                continue
            if identity not in reports:
                reports[identity] = format_mits_report(row) | {
                    "sources": [],
                    "field_sources": {field: [] for field in REPORT_FIELDS},
                }
            report = reports[identity]
            if source not in report["sources"]:
                report["sources"].append(source)
            for field, value in zip(REPORT_FIELDS, identity, strict=True):
                if value and source not in report["field_sources"][field]:
                    report["field_sources"][field].append(source)
    # Ancillary exports repeat the narrative without HEText. Merge this repetition only
    # when exactly one microscopy variant has the same specimen, diagnosis and comments.
    for identity in list(reports):
        if identity[2]:
            continue
        matches = [
            key
            for key in reports
            if key[2] and (key[0], key[1], key[3]) == (identity[0], identity[1], identity[3])
        ]
        if len(matches) != 1:
            continue
        repeated, target = reports.pop(identity), reports[matches[0]]
        target["sources"] = list(dict.fromkeys(target["sources"] + repeated["sources"]))
        for field in REPORT_FIELDS:
            target["field_sources"][field] = list(
                dict.fromkeys(target["field_sources"][field] + repeated["field_sources"][field])
            )
    return list(reports.values())


def case_data(case, filename=None):
    if not state["ready"]:
        return {"ready": False, "error": state["error"]}

    deid = D["study2deid"].get(case)
    out = {"ready": True, "case": case, "deid": deid, "slide": None}
    if filename:
        out["slide"] = D["slide_labels"].get((case, _stem(filename)))
        out["slide_meta"] = slide_meta(case, filename)

    if deid:
        out.update(_case_tables(deid))
    return out


def pathology_exports():
    """All records from each loaded source, including repeated specimen/test context."""
    return [
        {
            "title": title,
            "source": D["sources"]["cpl_" + key],
            "cases": D["cpl_" + key],
        }
        for key, title in (
            ("Diagnosis", "Pathology diagnosis"),
            ("IHC", "Immunohistochemistry"),
            ("SpecialStains", "Special stains"),
            ("PCR", "PCR"),
        )
    ]


# ---------------------------------------------------------------- disease map
def _decode_diseases(deid, measure):
    rec = D["decode_results"].get(deid)
    if not rec:
        return None

    if measure == "uc":
        groups = {rec.get("UC_champs_group_desc")}
    elif measure == "ic":
        groups = {rec.get("IC_champs_group_desc")}
    else:  # "any": every group in the causal chain
        groups = {v for k, v in rec.items() if k.endswith("champs_group_desc")}
    groups.discard(None)
    return groups or None


def _tac_diseases(deid):
    rec = D["tac_results"].get(deid)
    if not rec:
        return None

    out = set()
    for col, val in rec.items():
        m = TAC_RE.match(col)
        if val == "Positive" and m:
            name = D["tac_names"].get(m.group("code"), m.group("code"))
            out.add(name.split(" · ")[0])
    return out


def case_diseases(deid, measure):
    """Set of disease labels for one case under the chosen measure (None = not assessable)."""
    if measure in ("uc", "ic", "any"):
        return _decode_diseases(deid, measure)
    if measure == "tac":
        return _tac_diseases(deid)
    return None


def _count_sites(measure, deids, pathology):
    sites = {}
    for deid, demo in D["basic_demographics"].items():
        if deids is not None and deid not in deids:
            continue

        site = demo.get("site_iso_code")
        diseases = (
            (pathology.get(deid) or None)
            if measure == "pathology"
            else case_diseases(deid, measure)
        )
        if not site or diseases is None:
            continue

        s = sites.setdefault(site, {"n": 0, "counts": {}})
        s["n"] += 1
        for d in diseases:
            s["counts"][d] = s["counts"].get(d, 0) + 1
    return sites


def disease_map(measure="uc", deids=None, pathology=None):
    """Per-site case counts by disease. deids restricts to a set of cases (e.g. those with slides)."""
    sites = _count_sites(measure, deids, pathology or {})

    totals = {label: 0 for label, _ in CONDITIONS.values()} if measure == "pathology" else {}
    for s in sites.values():
        for d, n in s["counts"].items():
            totals[d] = totals.get(d, 0) + n

    return {
        "measure": measure,
        "sites": sites,
        "diseases": sorted(totals.items(), key=lambda kv: -kv[1]),
        "n_cases": sum(s["n"] for s in sites.values()),
    }
