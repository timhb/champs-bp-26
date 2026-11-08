"""Case joins must preserve every image and report without linking unknown IDs."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as viewer  # noqa: E402
import clinical  # noqa: E402
from pathology import CONDITIONS, case_catalog, slide_finding_index  # noqa: E402


def slide(study, number=0):
    return {"case": study, "country": "Test", "path": f"Test/Cases/{study}/{number}.svs"}


class CasePathologyTests(unittest.TestCase):
    def setUp(self):
        case_tables = patch.object(
            clinical, "_case_tables", return_value={"demographics": {"site_iso_code": "BD"}}
        )
        case_tables.start()
        self.addCleanup(case_tables.stop)
        self.mapping = {"S1": "D1", "S2": "D1", "REPORT_ONLY": "D2"}
        self.rows = [{"Specimen Info": "lung", "Diagnosis": "<p>Finding</p>"}] * 2
        self.reports = [
            {
                "title": "Diagnosis",
                "source": "diagnosis.csv",
                "cases": {"D1": self.rows, "D2": [{"Diagnosis": "Report only"}], "": [{}]},
            }
        ]

    def test_groups_study_aliases_and_keeps_all_records(self):
        slides = [slide("S1", n) for n in range(1200)] + [slide("S2"), slide("UNKNOWN")]
        cases = case_catalog(slides, self.mapping, self.reports)
        self.assertEqual(len(cases["deid:D1"]["slides"]), 1201)
        self.assertEqual(cases["deid:D1"]["studies"], ["S1", "S2"])
        self.assertEqual(cases["deid:D1"]["reports"][0]["records"], self.rows)
        self.assertEqual(cases["deid:D2"]["slides"], [])
        self.assertEqual(cases["study:UNKNOWN"]["reports"], [])
        self.assertNotIn("deid:", cases)

    def test_missing_folder_does_not_merge_unrelated_images(self):
        cases = case_catalog([slide("", 1), slide("", 2)], {}, [])
        self.assertEqual(len(cases), 2)

    def test_blank_report_ids_are_ignored(self):
        import pandas as pd

        table = pd.DataFrame({"champs_deid": [None, "", "  ", "D1"], "text": ["x"] * 4})
        self.assertEqual(clinical._by_deid(table), {"D1": [{"text": "x"}]})

    def test_mits_narratives_merge_repeated_fields_but_keep_amendments_and_provenance(self):
        shared = {
            "Specimen Info": "MITS",
            "Diagnosis": "Liver: autolysis",
            "DiagnosisComment": "Pending",
        }
        full = shared | {"HEText": "SITE REPORT MAJOR FINDINGS:\nLiver: autolysis"}
        amended = shared | {"DiagnosisComment": "Amended result"}
        groups = {
            "cpl_Diagnosis": {"D1": [full]},
            "cpl_IHC": {"D1": [shared, shared]},
            "cpl_PCR": {"D1": [amended]},
            "cpl_SpecialStains": {},
            "sources": {
                "cpl_Diagnosis": "diagnosis.csv",
                "cpl_IHC": "ihc.csv",
                "cpl_PCR": "pcr.csv",
            },
        }
        with patch.dict(clinical.D, groups):
            reports = clinical._mits_reports("D1")
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0]["sources"], ["diagnosis.csv", "ihc.csv"])
        self.assertEqual(reports[0]["field_sources"]["HEText"], ["diagnosis.csv"])
        self.assertEqual(reports[1]["original"]["DiagnosisComment"], "Amended result")

    def test_api_case_lookup_pagination_readiness_and_read_only(self):
        slides = [slide("S1", n) for n in range(1200)]
        with (
            patch.dict(viewer.index, {"ready": True, "files": slides, "enriched": True}),
            patch.dict(clinical.state, {"ready": True, "error": None}),
            patch.dict(clinical.D, {"study2deid": self.mapping, "decode_results": {}}),
            patch.object(clinical, "pathology_exports", return_value=self.reports),
        ):
            client = viewer.app.test_client()
            case = client.get("/api/pathology/case/study:S2").get_json()
            self.assertEqual(case["key"], "deid:D1")
            self.assertEqual(case["clinical"]["deid"], "D1")
            self.assertEqual(case["clinical"]["demographics"]["site_iso_code"], "BD")
            self.assertEqual(len(case["slides"]), 1200)
            self.assertEqual(len(case["reports"][0]["records"]), 2)
            self.assertEqual(case["reports"][0]["records"][0]["Diagnosis"], "Finding")
            # Display cleaning must not mutate the loaded source records.
            self.assertEqual(self.rows[0]["Diagnosis"], "<p>Finding</p>")
            found = client.get("/api/pathology/cases?q=REPORT_ONLY").get_json()
            self.assertEqual(found["total"], 1)
            self.assertEqual(found["items"][0]["slide_count"], 0)
            page = client.get("/api/pathology/cases?limit=1&offset=1").get_json()
            self.assertEqual(len(page["items"]), 1)
            self.assertEqual(client.get("/api/pathology/cases?limit=oops").status_code, 400)
            self.assertEqual(client.get("/api/pathology/case/deid:missing").status_code, 404)
            self.assertEqual(client.post("/api/pathology/case/deid:D1").status_code, 405)
            with client.get("/pathology") as page_response:
                self.assertEqual(page_response.status_code, 200)
            with patch.dict(clinical.state, {"ready": False}):
                self.assertFalse(client.get("/api/pathology/cases").get_json()["ready"])

    def test_finding_index_keeps_exact_positive_labels_and_organ_scope(self):
        records = [
            {
                "wsi_path": "data/Test/Cases/S1/0.svs",
                "organ": "right_lung",
                "semantic_groups": "aspiration_squames,aspiration_meconium,bronchopneumonia",
            },
            {
                "wsi_path": "/dataset/Test/Cases/S1/0.svs",
                "organ": "left_lung",
                "semantic_groups": "pneumonitis,hyaline_membranes,no_inflammation",
            },
            {
                "wsi_path": "data/Test/Cases/S1/1.svs",
                "organ": "liver",
                "semantic_groups": "steatosis,hemozoin_pigment,bronchopneumonia",
            },
            {
                "wsi_path": "data/Test/Cases/S1/2.svs",
                "organ": "lung",
                "semantic_groups": "no_bronchopneumonia,possible_pneumonitis,steatosis",
            },
            {"wsi_path": None, "organ": "lung", "semantic_groups": "bronchopneumonia"},
        ]
        findings = slide_finding_index(records, "/dataset")
        self.assertEqual(
            findings[slide("S1")["path"]],
            {name for name, organs in CONDITIONS.values() if "lung" in organs},
        )
        self.assertEqual(findings[slide("S1", 1)["path"]], {"Steatosis", "Hemozoin pigment"})
        self.assertNotIn(slide("S1", 2)["path"], findings)
        # Labels for unavailable images must not make a case match.
        cases = case_catalog([slide("S1", 2)], self.mapping, self.reports, slide_findings=findings)
        self.assertEqual(cases["deid:D1"]["conditions"], [])

    def test_condition_filters_combine_before_pagination_and_ignore_final_cod(self):
        deid = "024A1F57-EBE6-43F4-90ED-32AAC5648507"
        reports = self.reports + [
            {
                "title": "Diagnosis",
                "source": "diagnosis.csv",
                "cases": {deid: [{"Diagnosis": "Bronchopneumonia"}]},
            }
        ]
        demographics = {"D1": {"site_iso_code": "BD"}, deid: {"site_iso_code": "BD"}}
        # CoD contradicts the tissue findings, and report-only mentions are not image labels.
        decode = {
            "D1": {"UC_champs_group_desc": "Other"},
            deid: {"UC_champs_group_desc": "Bronchopneumonia"},
        }
        findings = {slide("S1")["path"]: {name for name, _ in CONDITIONS.values()}}
        with (
            patch.dict(viewer.index, {"ready": True, "files": [slide("S1")], "enriched": True}),
            patch.dict(clinical.state, {"ready": True, "error": None}),
            patch.dict(
                clinical.D,
                {
                    "study2deid": self.mapping,
                    "basic_demographics": demographics,
                    "decode_results": decode,
                    "slide_findings": findings,
                },
            ),
            patch.object(clinical, "pathology_exports", return_value=reports),
            patch.object(clinical, "case_diseases") as cod,
        ):
            client = viewer.app.test_client()
            result = client.get(
                "/api/pathology/cases?country=Bangladesh&slides=no&limit=1"
            ).get_json()
            self.assertEqual(result["total"], 1)
            case = result["items"][0]
            self.assertEqual(case["case_id"], "AAC5648507")
            self.assertEqual(case["conditions"], [])
            self.assertEqual(result["conditions"], [name for name, _ in CONDITIONS.values()])
            self.assertIn("Bangladesh", result["countries"])
            for condition, _ in CONDITIONS.values():
                result = client.get(
                    "/api/pathology/cases",
                    query_string={
                        "country": "Bangladesh",
                        "condition": condition,
                        "slides": "yes",
                        "limit": 1,
                    },
                ).get_json()
                self.assertEqual(result["total"], 1)
                self.assertEqual(result["items"][0]["key"], "deid:D1")
                self.assertNotIn("causes", result["items"][0])
            self.assertEqual(
                client.get(
                    "/api/pathology/cases",
                    query_string={
                        "condition": "Bronchopneumonia",
                        "slides": "no",
                    },
                ).get_json()["total"],
                0,
            )
            self.assertEqual(client.get("/api/pathology/cases?q=AAC5648507").get_json()["total"], 1)
            detail = client.get(f"/api/pathology/case/deid:{deid}").get_json()
            self.assertEqual(detail["case_id"], "AAC5648507")
            for query in ("slides=invalid", "condition=invalid"):
                self.assertEqual(client.get("/api/pathology/cases?" + query).status_code, 400)
            with client.get("/pathology") as page:
                html = page.get_data(as_text=True)
                self.assertIn('id="condition"', html)
                self.assertNotIn('id="cause"', html)
                self.assertNotIn('id="measure"', html)
            cod.assert_not_called()


if __name__ == "__main__":
    unittest.main()
