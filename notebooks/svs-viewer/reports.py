"""Conservative, lossless presentation structure for the CHAMPS MITS report exports."""

import html
import re

REPORT_FIELDS = ("Specimen Info", "Diagnosis", "HEText", "DiagnosisComment")
SECTION_TITLES = {
    "diagnosis": "Diagnosis",
    "site_findings": "SITE report findings",
    "site_review": "CPL review of SITE scans",
    "cpl_slides": "CPL slides",
    "microscopy": "Microscopy — source not specified",
    "comments": "Interpretation and comments",
    "clinical_history": "Reported clinical history",
    "tac": "Reported TAC results",
    "laboratory": "Reported laboratory results",
    "site_report": "SITE report availability / notes",
    "telepathology": "Telepathology discussion",
    "gross": "Gross description",
    "adequacy": "Reported technical adequacy criteria",
    "other": "Additional report text — source not classified",
}
HEADER_PATTERNS = [
    ("site_findings", r"SITE(?: PATH(?:OLOGY)?)? REPORT MAJOR FINDINGS"),
    ("site_review", r"CPL REVIEW OF SITE (?:SCAN{1,2}ED )?SLIDES"),
    ("cpl_slides", r"CPL (?:SLIDES(?: REVIEW)?|SLIDE REVIEW)"),
    ("site_report", r"SITE PATH(?:OLOGY)? REPORT"),
    ("tac", r"TAC(?: RESULTS)?(?: \([^\n]*\))?"),
    ("laboratory", r"LABORATORY (?:RESULTS|DATA(?: REPORTED BY SUBMITTER)?)(?: \([^\n]*\))?"),
    ("telepathology", r"TELEPATHOLOGY (?:DISCUSSION|NOTES)"),
    ("gross", r"GROSS DESCRIPTION|MACROSCOPIC EXAMINATION"),
    ("microscopy", r"MICROSCOPIC EXAMINATION"),
    ("clinical_history", r"CLINICAL HISTORY(?: AND MATERNAL HISTORY)?"),
    ("adequacy", r"TECHNICAL ADEQUACY"),
]
HEADERS = [(kind, re.compile(pattern, re.I)) for kind, pattern in HEADER_PATTERNS]

LUNG_PAIR = r"(?:right|r\.?)\s*(?:and|&|/)\s*(?:left|l\.?)\s+lungs?"
CNS_PAIR = r"CNS\s*,?\s*posterior\s*(?:and|&|/)\s*transnasal"
ORGAN_LABEL = (
    rf"(?:{LUNG_PAIR}|{CNS_PAIR}|(?:right|left|r\.?|l\.?)\s+lungs?"
    r"|lungs?|CNS\s*,?\s*(?:posterior|transnasal)|(?:posterior|transnasal)\s+CNS"
    r"|CNS|brain|liver|heart|spleen|kidneys?|placenta|adrenals?|bone marrow)"
)
ORGAN_LINE = re.compile(rf"^\s*(?P<label>{ORGAN_LABEL})\s*:\s*(?P<text>.*)$", re.I)
ORGAN_BREAK = re.compile(r"(?<=;)[ \t]*(?=[A-Za-z][A-Za-z ,./&()-]{0,80}:)")


def clean_report_text(value):
    if not isinstance(value, str):
        return ""
    value = value.replace("_x000D_", "")
    value = re.sub(r"(?i)<br\s*/?>|</div>|</p>|</li>|</tr>", "\n", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = html.unescape(value).replace("\xa0", " ")
    value = re.sub(r"[ \t]+\n", "\n", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def organ_keys(label):
    """Only normalize explicit organ/site labels; never infer an organ from findings."""
    label = re.sub(r"\s+", " ", label.lower()).strip()
    if re.fullmatch(LUNG_PAIR, label, re.I):
        return ["right_lung", "left_lung"]
    if re.fullmatch(CNS_PAIR, label, re.I):
        return ["cns_post", "cns_trans"]
    if re.fullmatch(r"(?:right|r\.?) lungs?", label):
        return ["right_lung"]
    if re.fullmatch(r"(?:left|l\.?) lungs?", label):
        return ["left_lung"]
    if "cns" in label and "posterior" in label:
        return ["cns_post"]
    if "cns" in label and "transnasal" in label:
        return ["cns_trans"]
    return [{"lungs": "lung", "brain": "cns", "kidneys": "kidney"}.get(label, label)]


def organ_entries(text):
    entries, notes = [], []
    current = None
    for line in text.splitlines():
        for part in ORGAN_BREAK.split(line):
            match = ORGAN_LINE.fullmatch(part)
            if match:
                current = {
                    "site": match["label"].strip(),
                    "organs": organ_keys(match["label"]),
                    "text": match["text"],
                }
                entries.append(current)
            elif re.match(r"^\s*[^:]{1,100}:\s*", part):
                # An unfamiliar tissue/header must not inherit the previous organ's scope.
                current = None
                notes.append(part)
            elif current is not None:
                current["text"] += "\n" + part
            else:
                notes.append(part)
    for entry in entries:
        entry["text"] = entry["text"].strip()
    return entries, "\n".join(notes).strip()


def _heading(line):
    label, sep, _ = line.partition(":")
    normalized = re.sub(r"\s+", " ", label.strip())
    for kind, pattern in HEADERS:
        if pattern.fullmatch(normalized):
            return kind, line.index(":") + 1 if sep else len(line)
    if (
        sep
        and len(normalized) > 3
        and normalized.isupper()
        and not ORGAN_LINE.fullmatch(line.strip())
    ):
        return "other", 0
    # These are complete statements rather than headings: keep the whole statement.
    for prefix, kind in (
        (r"technical adequacy\b", "adequacy"),
        (r"CPL slides? (?:not|were not) (?:performed|examined|reviewed)\b", "cpl_slides"),
        (r"clinical history (?:was|were) not provided\b", "clinical_history"),
    ):
        if re.match(prefix, line.strip(), re.I):
            return kind, 0
    return None


def split_report_field(text, field):
    default = {"Diagnosis": "diagnosis", "HEText": "microscopy", "DiagnosisComment": "comments"}
    markers = [(0, 0, default[field])]
    position = 0
    for line in text.splitlines(keepends=True):
        heading = _heading(line)
        if heading:
            kind, body_offset = heading
            if position == 0:
                markers.clear()
            markers.append((position, position + body_offset, kind))
        position += len(line)
    sections = []
    for index, (start, body_start, kind) in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else len(text)
        raw, body = text[start:end], text[body_start:end].strip()
        if not raw:
            continue
        if kind in ("diagnosis", "site_findings", "site_review", "cpl_slides", "microscopy"):
            entries, notes = organ_entries(body)
        else:
            entries, notes = [], body
        sections.append(
            {
                "kind": kind,
                "title": SECTION_TITLES[kind],
                "field": field,
                "raw": raw,
                "text": body,
                "entries": entries,
                "notes": notes,
            }
        )
    return sections


def format_mits_report(record):
    original = {key: clean_report_text(record.get(key)) for key in REPORT_FIELDS}
    return {
        "specimen": original["Specimen Info"],
        "mits_mentioned": bool(
            re.search(r"\bMITS\b|minimally invasive", "\n".join(original.values()), re.I)
        ),
        "sections": [
            section
            for field in REPORT_FIELDS[1:]
            for section in split_report_field(original[field], field)
        ],
        "original": original,
    }
