"""Report presentation must preserve source scope, wording, and nonstandard text."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reports import REPORT_FIELDS, format_mits_report  # noqa: E402


class MitsReportTests(unittest.TestCase):
    def test_site_review_and_cpl_tissue_stay_separate(self):
        record = {
            "Diagnosis": "Multiple tissues, minimally invasive tissue sampling:\n"
            "Right  and left lung: autolysis; rare squames.\n"
            "CNS, posterior and transnasal: site scanned slides not available to be reviewed.",
            "HEText": "SITE REPORT MAJOR FINDINGS:\nLiver: Severe autolysis\n"
            "CNS, posterior: unremarkable\n"
            "CPL REVIEW OF SITE SCANED SLIDES:\nLiver: autolysis\n"
            "CNS, posterior: site scanned slides not available to be reviewed.\n"
            "CPL SLIDES:\nNot performed",
            "DiagnosisComment": "Findings do not point to an infectious etiology.\n"
            "TAC results: Lung: negative; Blood: Escherichia coli/Shigella spp\n"
            "Laboratory results: Pending TB.\nTELEPATHOLOGY DISCUSSION\nNot performed\n"
            "Gross description: A photograph is provided.\n"
            "Technical adequacy requires the presence of at least 2 full cores of target tissue.",
        }
        report = format_mits_report(record)
        by_kind = {section["kind"]: section for section in report["sections"]}
        self.assertTrue(report["mits_mentioned"])
        self.assertEqual(by_kind["diagnosis"]["entries"][0]["organs"], ["right_lung", "left_lung"])
        self.assertEqual(by_kind["diagnosis"]["entries"][1]["organs"], ["cns_post", "cns_trans"])
        self.assertEqual(by_kind["site_findings"]["entries"][1]["text"], "unremarkable")
        self.assertIn("not available", by_kind["site_review"]["entries"][1]["text"])
        self.assertEqual(by_kind["cpl_slides"]["text"], "Not performed")
        self.assertIn("Lung: negative", by_kind["tac"]["text"])
        self.assertEqual(by_kind["laboratory"]["text"], "Pending TB.")
        self.assertIn("at least 2 full cores", by_kind["adequacy"]["text"])
        self.assertEqual(by_kind["telepathology"]["text"], "Not performed")
        self.assert_preserved(report)

    def assert_preserved(self, report):
        for field in REPORT_FIELDS[1:]:
            reconstructed = "".join(s["raw"] for s in report["sections"] if s["field"] == field)
            self.assertEqual(reconstructed, report["original"][field])

    def test_unknown_heading_is_retained_without_assigning_to_previous_organ(self):
        report = format_mits_report(
            {"HEText": "Liver: autolysis\nThymus: unusual finding\ncontinued"}
        )
        section = report["sections"][0]
        self.assertEqual(section["kind"], "microscopy")
        self.assertEqual(section["entries"][0]["text"], "autolysis")
        self.assertIn("Thymus: unusual finding\ncontinued", section["notes"])
        self.assert_preserved(report)

    def test_inline_organs_continuations_and_absent_mits_assertion(self):
        report = format_mits_report(
            {"Diagnosis": "Liver: no inflammation; R lung: rare squames\nwithout inflammation"}
        )
        self.assertFalse(report["mits_mentioned"])
        rows = report["sections"][0]["entries"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["organs"], ["right_lung"])
        self.assertIn("without inflammation", rows[1]["text"])
        self.assert_preserved(report)

    def test_missing_sections_and_html_have_safe_fallback(self):
        report = format_mits_report(
            {"Diagnosis": "<p>Unusual narrative</p><br>Uncertain.", "HEText": None}
        )
        self.assertEqual(len(report["sections"]), 1)
        self.assertIn("Unusual narrative", report["sections"][0]["notes"])
        self.assertFalse(report["mits_mentioned"])
        self.assert_preserved(report)

    def test_statement_about_other_cpl_tissues_does_not_become_an_organ_finding(self):
        report = format_mits_report(
            {
                "HEText": "CPL SLIDES:\nLiver: no target tissue\nCPL slides not examined for other tissues."
            }
        )
        self.assertEqual(len(report["sections"]), 2)
        self.assertEqual(report["sections"][0]["entries"][0]["text"], "no target tissue")
        self.assertIn("not examined", report["sections"][1]["text"])
        self.assert_preserved(report)

    def test_unrecognized_heading_does_not_inherit_cpl_source(self):
        report = format_mits_report(
            {"HEText": "CPL SLIDES:\nLIVER: autolysis\nUNUSUAL SOURCE:\nLiver: uncertain."}
        )
        self.assertEqual(report["sections"][0]["entries"][0]["text"], "autolysis")
        self.assertEqual(report["sections"][1]["kind"], "other")
        self.assertIn("UNUSUAL SOURCE", report["sections"][1]["text"])
        self.assert_preserved(report)


if __name__ == "__main__":
    unittest.main()
