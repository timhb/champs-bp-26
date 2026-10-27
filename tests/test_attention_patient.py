"""Case identity and organ scope must survive the patient-panel integration."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from bp26.attention.patient import PatientStore, load_backend, organ_context, scoped_reports
from bp26.attention.server import create_app


class PatientTests(unittest.TestCase):
    def setUp(self):
        self.row = SimpleNamespace(
            id="slide1",
            organ="lung",
            site="Site",
            slide_id="slide",
            wsi_path="/data/Site/Cases/STUDY1/slide.svs",
        )
        self.payload = dict(
            ready=True,
            case="STUDY1",
            deid="D1",
            slide=dict(organ="right_lung", source="SITE"),
            demographics=dict(sex="Female", age_days="0"),
            adequacy=[
                dict(organ="lung", subsite="right", adequate="false"),
                dict(organ="lung", subsite="left", adequate="true"),
                dict(organ="liver", adequate="true"),
            ],
            tac=[
                dict(specimen="Lung", result="Positive"),
                dict(specimen="Blood", result="Negative"),
                dict(specimen="CSF", result="Indeterminate"),
            ],
            lab=[],
            mits_reports=[
                dict(
                    specimen="Multiple organs",
                    mits_mentioned=True,
                    sources=["dx.csv"],
                    sections=[
                        dict(
                            kind="site_review",
                            title="CPL review of SITE scans",
                            field="HEText",
                            text="full",
                            raw="full",
                            entries=[
                                dict(organs=["right_lung"], site="Right lung", text="RIGHT"),
                                dict(organs=["left_lung"], site="Left lung", text="LEFT"),
                                dict(organs=["liver"], site="Liver", text="LIVER"),
                            ],
                            notes="Unscoped note",
                        ),
                        dict(
                            kind="cpl_slides",
                            title="CPL slides",
                            entries=[dict(organs=["lung"], text="CPL ONLY")],
                            notes="",
                        ),
                    ],
                )
            ],
        )
        self.catalog = pd.DataFrame([vars(self.row)])
        self.backend = SimpleNamespace(
            LEVEL2_TABLES=[
                "basic_demographics",
                "decode_results",
                "lab_results",
                "tac_results",
                "verbal_autopsy",
            ],
            D={
                "study2deid": {"STUDY1": "D1"},
                "sources": {
                    f"cpl_{key}": f"{key}.csv"
                    for key in ("Diagnosis", "IHC", "SpecialStains", "PCR")
                },
            },
            case_data=lambda study, filename: self.payload,
            _clean_text=lambda value: value if isinstance(value, str) else "",
        )
        for key in self.backend.LEVEL2_TABLES:
            self.backend.D[key] = {"D1": {"recorded_zero": "0", "value": key}}
        for key in ("Diagnosis", "IHC", "SpecialStains", "PCR"):
            self.backend.D[f"cpl_{key}"] = {
                "D1": [
                    dict(specimen="lung", result="one"),
                    dict(specimen="liver", result="two"),
                    dict(specimen="lung", result="one"),
                ]
            }

    def store(self):
        store = PatientStore(dict(data_root=Path("/data")), self.catalog)
        store.started = True
        store.backend = self.backend
        return store

    def test_side_and_report_sources_remain_separate(self):
        context = organ_context(self.payload, "lung")
        reports = scoped_reports(self.payload["mits_reports"], context)
        review, cpl = reports[0]["sections"]
        self.assertEqual([r["text"] for r in review["entries"]], ["RIGHT"])
        self.assertEqual(cpl["entries"][0]["text"], "CPL ONLY")
        self.assertEqual(review["notes"], "Unscoped note")
        self.assertNotIn("raw", review)
        self.assertEqual(len(self.payload["mits_reports"][0]["sections"][0]["entries"]), 3)

    def test_all_patient_records_retained_but_context_filtered(self):
        result = self.store().get(self.row)
        self.assertEqual(len(result["organ_adequacy"]), 1)
        self.assertEqual(result["organ_adequacy"][0]["subsite"], "right")
        self.assertEqual([r["specimen"] for r in result["related_tac"]], ["Lung", "Blood"])
        self.assertEqual(len(result["case"]["tac"]), 3)
        self.assertEqual(len(result["reports"][1]["records"]), 3)
        self.assertEqual(result["raw_tables"]["verbal_autopsy"]["recorded_zero"], "0")

    def test_missing_mapping_does_not_join_anonymous_records(self):
        self.payload = dict(ready=True, case="STUDY1", deid=None, slide=None)
        result = self.store().get(self.row)
        self.assertEqual(result["raw_tables"], {})
        self.assertEqual(result["reports"], [])
        self.assertEqual(result["organ_reports"], [])
        self.assertEqual(result["context"]["organ"], "lung")

    def test_shared_patient_can_have_multiple_study_ids(self):
        other = vars(self.row) | dict(id="slide2", wsi_path="/data/Site/Cases/STUDY2/other.svs")
        self.backend.D["study2deid"]["STUDY2"] = "D1"
        store = self.store()
        store.catalog = pd.DataFrame([vars(self.row), other])
        with patch("bp26.attention.patient.load_backend", return_value=self.backend):
            store._load()
        self.assertEqual(len(store.get(self.row)["available_slides"]), 2)

    def test_existing_svs_helpers_load_in_an_isolated_package(self):
        with patch("pandas.read_csv", side_effect=RuntimeError("fixture stops before data access")):
            with self.assertRaisesRegex(RuntimeError, "fixture stops"):
                load_backend(Path("/nonexistent"))

    def test_case_endpoint_is_scoped_to_catalog_and_not_cached(self):
        config = dict(
            data_root=Path("/data"),
            slide_registry=Path("/data/registry.parquet"),
            attention_cache_size=2,
            open_slide_cache_size=2,
        )
        catalog = self.catalog.set_index("id", drop=False)
        app = create_app(config, catalog)
        with patch.object(
            app.extensions["patient_store"], "get", return_value=dict(ready=False, error=None)
        ) as get:
            client = app.test_client()
            response = client.get("/api/slides/slide1/case")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertFalse(response.get_json()["ready"])
            self.assertEqual(client.get("/api/slides/unknown/case").status_code, 404)
            get.assert_called_once()
