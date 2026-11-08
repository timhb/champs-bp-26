"""Local medical links must preserve report text, phrase scope, and ambiguity."""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as viewer  # noqa: E402
from glossary import MedicalGlossary, mesh_records  # noqa: E402


def glossary():
    entries = [
        ("M1", "Necrosis", ["Necrosis"]),
        ("M2", "Edema", ["Edema"]),
        ("M3", "Pulmonary Edema", ["Pulmonary Edema"]),
        ("M4", "Inflammation", ["Inflammation", "Ambiguous"]),
        ("M5", "Another concept", ["Ambiguous"]),
        ("M6", "Tuberculosis", ["Tuberculosis", "TB"]),
    ]
    return MedicalGlossary(
        {
            "year": 2026,
            "records": [
                {
                    "id": uid,
                    "descriptor": "D001329",
                    "name": name,
                    "definition": name + " definition",
                    "tree_numbers": ["C01.100"],
                    "terms": terms,
                }
                for uid, name, terms in entries
            ],
        }
    )


class MedicalGlossaryTests(unittest.TestCase):
    def test_longest_phrase_and_whole_word_boundaries(self):
        source = "No inflammation. Pulmonary edema normal; micronecrosis is not necrosis."
        matches = glossary().matches(source)
        self.assertEqual(
            [source[m["start"] : m["end"]] for m in matches],
            ["inflammation", "Pulmonary edema", "necrosis"],
        )
        self.assertEqual(matches[1]["ids"], ["M3"])
        self.assertNotIn(
            "M3", [uid for m in glossary().matches("pulmonary; edema") for uid in m["ids"]]
        )

    def test_ambiguous_terms_do_not_pick_one_definition(self):
        match = glossary().matches("Ambiguous")[0]
        self.assertEqual(match["ids"], ["M4", "M5"])
        self.assertEqual([r["id"] for r in glossary().search("ambiguous")], ["M5", "M4"])

    def test_unicode_offsets_and_short_acronyms_preserve_text(self):
        source = "🔬 No necrosis. TB pending, tb is not automatically matched."
        matches = glossary().matches(source)
        self.assertEqual([source[m["start"] : m["end"]] for m in matches], ["necrosis", "TB"])
        fragments, cursor = [], 0
        for match in matches:
            fragments.extend(
                (source[cursor : match["start"]], source[match["start"] : match["end"]])
            )
            cursor = match["end"]
        fragments.append(source[cursor:])
        self.assertEqual("".join(fragments), source)

    def test_procedure_stain_and_normal_cells_are_not_automatically_linked(self):
        import json

        path = Path(__file__).resolve().parents[1] / "configs/medical_reference_terms.json"
        vocabulary = MedicalGlossary({"records": json.loads(path.read_text())})
        source = "MITS: HE, H&E staining and rare squames; he is a pronoun."
        self.assertEqual(
            [source[m["start"] : m["end"]] for m in vocabulary.matches(source)],
            [],
        )
        self.assertIn("cancer.gov", vocabulary.entry("NCI-HE")["url"])

    def test_only_conditions_are_linked_even_when_other_terms_are_available(self):
        import json

        vocabulary = MedicalGlossary(
            {
                "records": [
                    {
                        "id": name,
                        "name": name,
                        "terms": [name],
                        "tree_numbers": trees,
                        "descriptor": "D1",
                    }
                    for name, trees in [
                        ("liver", ["A03.620"]),
                        ("lung", ["A04.411"]),
                        ("blood", ["A12.207"]),
                        ("report", ["L01.100"]),
                        ("MITS", []),
                        ("pneumonia", ["C08.381.677"]),
                        ("depression", ["F03.600.300"]),
                        ("unclassified", []),
                    ]
                ]
            }
        )
        source = "MITS report: liver, lung, blood; pneumonia and depression; unclassified."
        self.assertEqual(
            [source[m["start"] : m["end"]] for m in vocabulary.matches(source)],
            ["pneumonia", "depression"],
        )
        # Broad manual lookup remains available without adding inline links.
        self.assertEqual(vocabulary.search("liver")[0]["name"], "liver")
        root = Path(__file__).resolve().parents[1]
        images = json.loads((root / "configs/medical_reference_images.json").read_text())
        for example in images.values():
            response = viewer.app.test_client().get(example["url"])
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "image/jpeg")
            response.close()
            self.assertTrue(example["caption"])
            self.assertTrue(example["source_url"].startswith("https://wwwn.cdc.gov/phil/"))

    def test_search_does_not_confuse_mits_with_limits_or_permits(self):
        vocabulary = MedicalGlossary(
            {
                "records": [
                    {"id": "M1", "name": "Limit of Detection", "terms": ["Detection limits"]},
                    {"id": "M2", "name": "Permits", "terms": ["Permits"]},
                ]
            }
        )
        self.assertEqual(vocabulary.search("MITS"), [])

    def test_xml_preserves_concept_specific_definitions_and_labels_broader_fallback(self):
        xml = b"""<DescriptorRecordSet><DescriptorRecord><DescriptorUI>D1</DescriptorUI>
          <DescriptorName><String>Parent heading</String></DescriptorName>
          <TreeNumberList><TreeNumber>C01.100</TreeNumber></TreeNumberList><ConceptList>
          <Concept PreferredConceptYN="Y"><ConceptUI>M1</ConceptUI>
          <ConceptName><String>Parent heading</String></ConceptName><ScopeNote>Parent definition.</ScopeNote>
          <TermList><Term><String>Parent alias</String></Term></TermList></Concept>
          <Concept><ConceptUI>M2</ConceptUI><ConceptName><String>Specific concept</String></ConceptName></Concept>
          </ConceptList></DescriptorRecord></DescriptorRecordSet>"""
        rows = list(mesh_records(io.BytesIO(xml)))
        self.assertEqual(rows[0]["tree_numbers"], ["C01.100"])
        self.assertEqual(rows[0]["definition"], "Parent definition.")
        self.assertIn("Parent alias", rows[0]["terms"])
        self.assertEqual(rows[1]["definition"], "")
        self.assertEqual(rows[1]["broader_definition"], "Parent definition.")

    def test_endpoints_use_local_data_and_validate_input(self):
        with (
            patch.object(viewer, "_glossary", return_value=glossary()),
            patch("urllib.request.urlopen") as external,
        ):
            client = viewer.app.test_client()
            response = client.post("/api/glossary/matches", json={"texts": ["No necrosis."]})
            self.assertEqual(response.get_json()["matches"][0][0]["ids"], ["M1"])
            self.assertEqual(client.get("/api/glossary/entry/M1").get_json()["name"], "Necrosis")
            self.assertEqual(
                client.get("/api/glossary/search?q=necrosis").get_json()["items"][0]["id"], "M1"
            )
            self.assertEqual(client.get("/api/glossary/status").get_json()["concepts"], 6)
            for payload in [
                None,
                [],
                {"texts": [42]},
                {"texts": ["x"] * 201},
                {"texts": ["x" * 30001]},
            ]:
                self.assertEqual(
                    client.post("/api/glossary/matches", json=payload).status_code, 400
                )
            self.assertEqual(client.get("/api/glossary/entry/missing").status_code, 404)
            self.assertEqual(client.get("/api/glossary/search?q=" + "x" * 121).status_code, 400)
            external.assert_not_called()


if __name__ == "__main__":
    unittest.main()
