"""Build a local medical glossary from official NLM MeSH descriptor XML."""

import argparse
import gzip
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from glossary import mesh_records

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "notebooks/svs-viewer/configs/svs_viewer.toml"
    )
    parser.add_argument(
        "--xml-gz", type=Path, help="Use an already downloaded official XML archive"
    )
    args = parser.parse_args()
    cfg = tomllib.loads(args.config.read_text())
    source = cfg["glossary_source_url"]
    with args.xml_gz.open("rb") if args.xml_gz else urlopen(source, timeout=60) as archive:
        with gzip.GzipFile(fileobj=archive) as stream:
            records = list(mesh_records(stream))
    additions = ROOT / cfg["glossary_additions_file"]
    records.extend(json.loads(additions.read_text()))
    images = json.loads((ROOT / cfg["glossary_images_file"]).read_text())
    for record in records:
        if record["id"] in images:
            record["histology_image"] = images[record["id"]]
    output = ROOT / cfg["glossary_file"]
    data = {
        "source": "U.S. National Library of Medicine — Medical Subject Headings (MeSH)",
        "source_url": source,
        "year": cfg["glossary_year"],
        "condition_links": True,
        "built": datetime.now(UTC).isoformat(),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False))
    temporary.replace(output)
    print(f"Saved {len(records):,} medical reference entries to {output}")


if __name__ == "__main__":
    main()
