"""Inventory source-report layouts without exporting case identifiers or report text."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from reports import REPORT_FIELDS, format_mits_report


def audit_file(path):
    counts, fields, sections = Counter(), Counter(), Counter()
    cases = set()
    with path.open() as handle:
        for row in csv.DictReader(handle):
            counts["rows"] += 1
            case = row.get("champs_deid", "").strip()
            if case:
                cases.add(case)
            else:
                counts["blank_case_ids"] += 1
            report = format_mits_report(row)
            counts["mits_explicitly_mentioned"] += report["mits_mentioned"]
            counts["rows_with_organ_entries"] += any(s["entries"] for s in report["sections"])
            counts["rows_with_recognized_headings"] += any(
                s["kind"] not in ("diagnosis", "microscopy", "comments", "other")
                for s in report["sections"]
            )
            fields.update(k for k, value in report["original"].items() if value)
            sections.update(sorted({s["kind"] for s in report["sections"]}))
            for field in REPORT_FIELDS[1:]:
                reconstructed = "".join(s["raw"] for s in report["sections"] if s["field"] == field)
                if reconstructed != report["original"][field]:
                    raise ValueError(
                        f"Report text was lost while parsing {path.name}, row {counts['rows']}"
                    )
    return dict(counts) | {
        "distinct_linkable_cases": len(cases),
        "fields_present": dict(fields),
        "rows_containing_section": dict(sections),
        "all_cleaned_text_preserved": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    files = sorted(args.reports.glob("deid_CPL_DB_*.csv"))
    if not files:
        parser.error("No CPL report exports found")
    result = {path.name: audit_file(path) for path in files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
