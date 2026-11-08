"""Pathology map counts and slide links must match the case pathology filter."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as viewer  # noqa: E402
import clinical  # noqa: E402
from pathology import CONDITIONS  # noqa: E402


class PathologyMapTests(unittest.TestCase):
    def test_map_reuses_case_conditions_and_slide_filter(self):
        files = [
            {"case": study, "deid": deid, "country": "Bangladesh", "path": path}
            for study, deid, path in (
                ("S1", "D1", "one.svs"),
                ("S2", "D1", "two.svs"),
                ("S3", "D2", "three.svs"),
                ("S4", "D3", "four.svs"),
            )
        ]
        reports = [{"title": "Diagnosis", "source": "test", "cases": {"D4": [{}]}}]
        with (
            patch.dict(viewer.index, {"ready": True, "files": files, "enriched": True}),
            patch.dict(clinical.state, {"ready": True, "error": None}),
            patch.dict(
                clinical.D,
                {
                    "study2deid": {f["case"]: f["deid"] for f in files},
                    "basic_demographics": {
                        deid: {"site_iso_code": "BD"} for deid in ("D1", "D2", "D3", "D4")
                    },
                    "slide_findings": {
                        "one.svs": {"Bronchopneumonia", "Steatosis"},
                        "two.svs": {"Bronchopneumonia"},
                        "three.svs": {"Steatosis"},
                        "unavailable.svs": {"Pneumonitis"},
                    },
                    "decode_results": {"D3": {"UC_champs_group_desc": "Pneumonitis"}},
                },
            ),
            patch.object(clinical, "pathology_exports", return_value=reports),
        ):
            client = viewer.app.test_client()
            result = client.get("/api/disease_map?measure=pathology").get_json()
            self.assertTrue(result["ready"])
            self.assertEqual(result["n_cases"], 2)
            counts = dict(result["diseases"])
            self.assertEqual(set(counts), {label for label, _ in CONDITIONS.values()})
            self.assertEqual(counts["Bronchopneumonia"], 1)
            self.assertEqual(counts["Steatosis"], 2)
            self.assertEqual(counts["Pneumonitis"], 0)
            self.assertEqual(result["sites"]["BD"]["n"], 2)
            for scope in ("all", "slides"):
                scoped = client.get(f"/api/disease_map?measure=pathology&scope={scope}").get_json()
                self.assertEqual(scoped, result)
            for condition, count in counts.items():
                cases = client.get(
                    "/api/pathology/cases", query_string={"condition": condition}
                ).get_json()
                self.assertEqual(cases["total"], count)
            filtered = viewer._filter_files(
                files, {"cause": "Bronchopneumonia", "measure": "pathology"}
            )
            self.assertEqual([f["path"] for f in filtered], ["one.svs", "two.svs"])
            with patch.dict(viewer.index, {"ready": False}):
                pending = client.get("/api/disease_map?measure=pathology").get_json()
                self.assertFalse(pending["ready"])
            with client.get("/map") as page:
                self.assertIn('value="pathology"', page.get_data(as_text=True))
            self.assertEqual(client.get("/api/disease_map?measure=invalid").status_code, 400)


if __name__ == "__main__":
    unittest.main()
