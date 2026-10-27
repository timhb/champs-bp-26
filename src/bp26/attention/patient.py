"""Reuse the SVS viewer's read-only patient tables and lossless report parser."""

import importlib.util
import sys
import threading
from pathlib import Path

SVS = Path(__file__).resolve().parents[3] / "notebooks/svs-viewer"


def load_backend(root):
    # A private package isolates the SVS module's table state and relative imports.
    spec = importlib.util.spec_from_file_location(
        "bp26.attention._svs_clinical", SVS / "clinical.py", submodule_search_locations=[str(SVS)]
    )
    backend = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = backend
    spec.loader.exec_module(backend)
    backend.configure(str(root))
    backend.load()
    return backend


def organ_context(payload, selected_organ):
    organ = (payload.get("slide") or {}).get("organ") or selected_organ
    organs = {"right_lung": ("lung", "right"), "left_lung": ("lung", "left")}
    canonical, side = organs.get(organ, (organ, None))
    if canonical != selected_organ:
        organ, canonical, side = selected_organ, selected_organ, None
    source = (payload.get("slide") or {}).get("source")
    return dict(organ=organ, canonical=canonical, side=side, source=source or "Not recorded")


def matches_organ(entry, context):
    organs = entry.get("organs", [])
    return (
        context["organ"] in organs
        or context["canonical"] in organs
        or context["organ"] == "lung"
        and any(o in {"right_lung", "left_lung"} for o in organs)
    )


def scoped_reports(reports, context):
    """Filter explicit organ entries; preserve unscoped notes as report-level context."""
    result = []
    for report in reports:
        sections = []
        for section in report["sections"]:
            entries = [
                entry for entry in section.get("entries", []) if matches_organ(entry, context)
            ]
            selected = {k: v for k, v in section.items() if k not in {"entries", "raw", "text"}}
            selected["entries"] = entries
            # Unclassified text is never presented as a finding of the selected organ.
            selected["context_only"] = not bool(entries)
            sections.append(selected)
        result.append(
            {
                "specimen": report["specimen"],
                "mits_mentioned": report["mits_mentioned"],
                "sources": report["sources"],
                "sections": sections,
            }
        )
    return result


def related_specimen(specimen, organ):
    value = str(specimen or "").lower()
    return value.startswith("blood") or organ == "lung" and value == "lung"


class PatientStore:
    """Load once in a background thread, without delaying slide attention."""

    def __init__(self, config, catalog):
        self.root, self.catalog = config["data_root"], catalog
        self.backend, self.error, self.started = None, None, False
        self.lock = threading.Lock()
        self.case_slides = {}

    def start(self):
        with self.lock:
            if self.started:
                return
            self.started = True
            threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            backend = load_backend(self.root)
            cases = {}
            for row in self.catalog.itertuples():
                study = Path(row.wsi_path).parent.name
                deid = backend.D["study2deid"].get(study)
                key = ("deid", deid) if deid else ("study", study)
                cases.setdefault(key, []).append(
                    dict(
                        id=row.id,
                        slide_id=row.slide_id,
                        organ=row.organ,
                        site=row.site,
                        study_id=study,
                    )
                )
            self.case_slides = cases
            self.backend = backend
        except Exception as error:
            self.error = str(error)

    def get(self, row):
        self.start()
        backend = self.backend
        if backend is None:
            return dict(ready=False, error=self.error)
        path = Path(row.wsi_path)
        study = path.parent.name
        payload = backend.case_data(study, path.name)
        context = organ_context(payload, row.organ)
        deid = payload.get("deid")
        key = ("deid", deid) if deid else ("study", study)
        reports, raw_tables = [], {}
        if deid:
            for table in backend.LEVEL2_TABLES:
                raw_tables[table] = backend.D[table].get(deid) or {}
            for code, title in (
                ("Diagnosis", "Pathology diagnosis"),
                ("IHC", "Immunohistochemistry"),
                ("SpecialStains", "Special stains"),
                ("PCR", "PCR"),
            ):
                rows = backend.D["cpl_" + code].get(deid, [])
                reports.append(
                    dict(
                        title=title,
                        source=backend.D["sources"].get("cpl_" + code),
                        records=[
                            {k: backend._clean_text(v) for k, v in record.items()}
                            for record in rows
                        ],
                    )
                )
        adequacy = [
            r
            for r in payload.get("adequacy") or []
            if r.get("organ") == context["canonical"]
            and (not context["side"] or not r.get("subsite") or r["subsite"] == context["side"])
        ]
        return dict(
            ready=True,
            case=payload,
            context=context,
            raw_tables=raw_tables,
            reports=reports,
            organ_reports=scoped_reports(payload.get("mits_reports") or [], context),
            organ_adequacy=adequacy,
            related_tac=[
                r
                for r in payload.get("tac") or []
                if related_specimen(r.get("specimen"), row.organ)
            ],
            related_lab=[
                r
                for r in payload.get("lab") or []
                if related_specimen(r.get("specimen"), row.organ)
            ],
            available_slides=self.case_slides.get(key, []),
        )
