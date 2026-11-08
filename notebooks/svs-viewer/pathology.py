"""Group read-only pathology inputs by clinical case, preserving report records."""

CONDITIONS = {
    "aspiration_squames": ("Aspiration of squames", {"lung", "right_lung", "left_lung"}),
    "aspiration_meconium": ("Aspiration of meconium", {"lung", "right_lung", "left_lung"}),
    "bronchopneumonia": ("Bronchopneumonia", {"lung", "right_lung", "left_lung"}),
    "pneumonitis": ("Pneumonitis", {"lung", "right_lung", "left_lung"}),
    "hyaline_membranes": ("Hyaline membranes", {"lung", "right_lung", "left_lung"}),
    "steatosis": ("Steatosis", {"liver"}),
    "hemozoin_pigment": ("Hemozoin pigment", {"liver"}),
}


def slide_finding_index(records, data_root):
    """Index report-derived positive tissue labels by exact image path, merging duplicates.

    These labels describe tissue bags, not a confirmed finding on every physical slide.
    Missing labels are unknown. Never infer findings from CoD or free-text mentions.
    """
    findings = {}
    for record in records:
        path = record.get("wsi_path")
        if not isinstance(path, str) or not path:
            continue
        path = path.removeprefix(data_root.rstrip("/") + "/").removeprefix("data/")
        groups = {g.strip() for g in (record.get("semantic_groups") or "").split(",")}
        for key, (label, organs) in CONDITIONS.items():
            if key in groups and record.get("organ") in organs:
                findings.setdefault(path, set()).add(label)
    return findings


COUNTRY_NAMES = {
    "BD": "Bangladesh",
    "ET": "Ethiopia",
    "KE": "Kenya",
    "ML": "Mali",
    "MZ": "Mozambique",
    "NG": "Nigeria",
    "PK": "Pakistan",
    "SL": "Sierra Leone",
    "ZA": "South Africa",
}


def case_key(study, study_to_deid):
    deid = study_to_deid.get(study)
    return f"deid:{deid}" if deid else f"study:{study}"


def case_catalog(slides, study_to_deid, reports, demographics=None, slide_findings=None):
    """Include image-only and report-only cases; never join blank identifiers."""
    cases = {}

    def entry(key, deid=None):
        return cases.setdefault(
            key,
            {"key": key, "deid": deid, "studies": [], "countries": [], "slides": [], "reports": []},
        )

    for study, deid in study_to_deid.items():
        if study and deid:
            entry(case_key(study, study_to_deid), deid)["studies"].append(study)

    for slide in slides:
        study = slide["case"]
        # Slides without a case folder must not be joined to one anonymous case.
        key = case_key(study, study_to_deid) if study else f"file:{slide['path']}"
        case = entry(key, study_to_deid.get(study))
        case["slides"].append(slide)
        if study and study not in case["studies"]:
            case["studies"].append(study)
        country = slide["country"].replace("_", " ")
        if country.startswith("Nigeria "):
            country = "Nigeria"
        if country and country not in case["countries"]:
            case["countries"].append(country)

    for report in reports:
        for deid, records in report["cases"].items():
            if not deid or not deid.strip() or not records:
                continue
            entry(f"deid:{deid}", deid)["reports"].append(
                {"title": report["title"], "source": report["source"], "records": records}
            )

    cases = {key: c for key, c in cases.items() if c["slides"] or c["reports"]}
    for case in cases.values():
        case["conditions"] = sorted(
            {
                condition
                for slide in case["slides"]
                for condition in (slide_findings or {}).get(slide["path"], ())
            }
        )
        case["case_id"] = (
            case["deid"][-10:] if case["deid"] else ", ".join(case["studies"]) or case["key"]
        )
        site = (demographics or {}).get(case["deid"], {}).get("site_iso_code")
        country = COUNTRY_NAMES.get(site, site)
        if country and country not in case["countries"]:
            case["countries"].append(country)
        if not case["countries"]:
            case["countries"] = ["Unknown"]
    return cases


def case_summary(case):
    return {k: v for k, v in case.items() if k not in ("slides", "reports")} | {
        "slide_count": len(case["slides"]),
        "report_count": sum(len(r["records"]) for r in case["reports"]),
    }
