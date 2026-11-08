"""Render the concise EDA report (ticket #8) as a PDF from the two running viewers.

Numbers come from the viewers' JSON APIs and the shared WSI inventory; figures are
headless-browser screenshots of the viewer pages. Start both viewers before running.
"""

import argparse
import html
import json
import subprocess
import time
import tomllib
import urllib.request
from datetime import date
from pathlib import Path
from string import Template
from urllib.parse import urlencode

import pandas as pd

from bp26.browser import chromium

REPO = Path(__file__).resolve().parents[2]
VIEWPORT = {"width": 1400, "height": 900}


def load_settings(path):
    with open(path, "rb") as stream:
        settings = tomllib.load(stream)
    settings["out_dir"] = REPO / settings["out_dir"]
    return settings


def get_json(url, timeout=1800):
    """GET JSON, polling while a viewer job reports status "computing"."""
    deadline = time.monotonic() + timeout
    while True:
        with urllib.request.urlopen(url, timeout=600) as response:
            data = json.load(response)
        busy = data.get("status") == "computing" or data.get("ready") is False
        if not busy:
            return data
        if time.monotonic() > deadline:
            raise TimeoutError(f"Viewer still busy after {timeout} s: {url}")
        time.sleep(3)


def inventory_summary(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["format"] = frame.wsi_path.str.rsplit(".", n=1).str[-1].str.lower()
    power = pd.to_numeric(frame.objective_power, errors="coerce")
    return dict(
        slides=len(frame),
        cases=frame.case_id.nunique(),
        sites=frame.site.str.replace("_", " ").value_counts(),
        formats=frame.format.value_counts(),
        power=power.map(lambda p: "unknown" if pd.isna(p) else f"{p:g}×").value_counts(),
        probe_errors=int(frame.probe_error.ne("").sum()),
    )


def svs_summary(base, top):
    facets = get_json(f"{base}/api/facets")
    while not facets["clinical_ready"]:
        if facets["clinical_error"]:
            raise RuntimeError(f"SVS viewer tables failed: {facets['clinical_error']}")
        time.sleep(3)
        facets = get_json(f"{base}/api/facets")
    causes = get_json(f"{base}/api/disease_map?measure=uc")
    return dict(facets=facets, causes=causes, top=causes["diseases"][:top])


def cluster_query(settings, catalog):
    query = dict(organ=settings["organ"], branch=settings["branch"])
    query["method"] = settings["cluster_method"]
    branch = catalog["organs"][0]["branches"][settings["branch"]]
    return query, branch


def embedding_summary(settings):
    base = settings["attention_viewer"]
    catalog = get_json(f"{base}/api/graph/catalog")
    catalog["organs"] = [o for o in catalog["organs"] if o["organ"] == settings["organ"]]
    query, branch = cluster_query(settings, catalog)
    clusters = get_json(f"{base}/api/graph/clusters?{urlencode(query)}")
    associations = get_json(
        f"{base}/api/graph/associations?{urlencode(query | dict(include_noise='false'))}"
    )
    return dict(catalog=catalog, branch=branch, clusters=clusters, associations=associations)


def shoot_union(page, selector, path, pad=24):
    """Screenshot the bounding box of every element matching selector."""
    box = page.evaluate(
        """selector => {
            const r = [...document.querySelectorAll(selector)].map(e => e.getBoundingClientRect())
                .filter(b => b.width && b.height);
            const x = Math.min(...r.map(b => b.left)), y = Math.min(...r.map(b => b.top));
            return {x: x + scrollX, y: y + scrollY, width: Math.max(...r.map(b => b.right)) - x,
                    height: Math.max(...r.map(b => b.bottom)) - y};
        }""",
        selector,
    )
    clip = dict(x=box["x"] - pad, y=box["y"] - pad)
    clip |= dict(width=box["width"] + 2 * pad, height=box["height"] + 2 * pad)
    page.screenshot(path=path, clip=clip, full_page=True)


def shoot_map(page, base, out):
    """Circle map of cases per site; returns the viewer's site names by code."""
    page.goto(f"{base}/map")
    page.wait_for_function("document.querySelector('#status').textContent.includes('categories')")
    page.wait_for_timeout(1000)
    shoot_union(page, "#map circle, #map text", out / "map-circles.png")
    return page.evaluate("Object.fromEntries(Object.entries(SITES).map(([k, v]) => [k, v.name]))")


def shoot_graph(page, settings, out):
    """Projection coloured by cluster and by site; waits out the viewer's slow jobs."""
    page.set_default_timeout(1_800_000)
    page.goto(f"{settings['attention_viewer']}/graph")
    page.wait_for_selector("#cluster-run:not([disabled])")
    page.select_option("#graph-branch", str(settings["branch"]))
    page.select_option("#graph-projection", settings["projection"])
    page.select_option("#cluster-method", settings["cluster_method"])
    page.click("#cluster-run")  # waits until the projection has loaded and the button re-enables
    page.wait_for_selector("#association-metrics > *")
    for color in ("cluster", "site"):
        page.select_option("#graph-color", color)
        page.wait_for_timeout(2000)
        shoot_union(page, ".graph-stage, #graph-legend", out / f"graph-{color}.png", pad=0)


def bars(series, total=None):
    """Static single-series horizontal bars with direct value labels."""
    total = total or series.sum()
    top = series.max()
    rows = "".join(
        f'<div class="bar"><span>{html.escape(str(label))}</span>'
        f'<div><i style="width:{100 * value / top:.1f}%"></i></div>'
        f"<b>{value:,}</b><small>{100 * value / total:.0f}%</small></div>"
        for label, value in series.items()
    )
    return f'<div class="bars">{rows}</div>'


def facet_series(rows, limit=8):
    """Viewer facets ([[name, count], ...]) as a Series, folding the tail into Other."""
    names = {"(NOT IN FILENAME)": "No stain token"}
    series = pd.Series({names.get(n, n.replace("_", " ")): c for n, c in rows})
    series = series.sort_values(ascending=False)
    if len(series) > limit:
        series = pd.concat([series[: limit - 1], pd.Series({"Other": series[limit - 1 :].sum()})])
    return series


def stacked(causes, names, k=8):
    """100% bars of each site's underlying causes; top k overall, in the map's colours."""
    top = [disease for disease, _ in causes["diseases"][:k]]
    colors = [f"var(--s{i + 1})" for i in range(k)] + ["var(--other)"]
    legend = "".join(
        f'<span><i style="background:{c}"></i>{html.escape(d)}</span>'
        for d, c in zip([*top, "Other"], colors, strict=True)
    )
    rows = []
    for code, site in sorted(causes["sites"].items(), key=lambda kv: -kv[1]["n"]):
        parts = [site["counts"].get(d, 0) for d in top]
        parts.append(site["n"] - sum(parts))
        segments = "".join(
            f'<i style="flex:{n};background:{c}"></i>'
            for n, c in zip(parts, colors, strict=True)
            if n
        )
        label = f"{html.escape(names.get(code, code))} <small>{site['n']:,}</small>"
        rows.append(f'<div class="stack"><span>{label}</span><div>{segments}</div></div>')
    return f'<div class="legend">{legend}</div>{"".join(rows)}'


def association_rows(associations):
    names = dict(site="Site", age_group="Age group", sex="Sex", age_months="Age (months)")
    metric = dict(cramers_v="Cramér's V", eta_squared="η²")
    rows = []
    for key, name in names.items():
        prop = associations["properties"][key]
        value = "n/a" if prop["value"] is None else f"{prop['value']:.3f}"
        ci = "" if not prop["ci"] else f"[{prop['ci'][0]:.3f}, {prop['ci'][1]:.3f}]"
        ami = "—" if prop["ami"] is None else f"{prop['ami']:.3f}"
        rows.append(
            f"<tr><td>{name}</td><td>{metric[prop['metric']]}</td><td>{value} {ci}</td>"
            f"<td>{ami}</td><td>{prop['known']:,}</td></tr>"
        )
    return "".join(rows)


def observations(inventory, causes, associations):
    """Three data-driven headline sentences."""
    formats = inventory["formats"] / inventory["slides"]
    cause, count = causes["diseases"][0]
    props = associations["properties"]
    labels = dict(site="site", age_group="age group", sex="sex")
    strongest = max(labels, key=lambda key: props[key]["value"] or 0)
    items = [
        f"{100 * formats.get('ndpi', 0):.0f}% of slides are <code>.ndpi</code>; the SVS viewer and its "
        f"stain and organ breakdowns cover only the {100 * formats.get('svs', 0):.0f}% in "
        "<code>.svs</code>.",
        f"{html.escape(cause)} is the underlying cause for {100 * count / causes['n_cases']:.0f}% "
        "of cases with a DeCoDe result (per-site mix in section 2).",
        f"Embedding clusters associate most with {labels[strongest]} "
        f"(Cramér's V {props[strongest]['value']:.2f}).",
    ]
    return "".join(f"<li>{item}</li>" for item in items)


def revision():
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()


def render(settings, inventory, svs, embedding, names):
    facets, causes = svs["facets"], svs["causes"]
    top = pd.Series(dict(svs["top"]))
    cohort = embedding["associations"]["cohort"]
    n_clusters = len({label for label in embedding["clusters"]["labels"] if label >= 0})
    organ = embedding["catalog"]["organs"][0]
    template = Template(Path(__file__).with_name("eda_report.html").read_text())
    values = dict(
        date=date.today().isoformat(),
        revision=revision(),
        slides=f"{inventory['slides']:,}",
        cases=f"{inventory['cases']:,}",
        n_sites=len(inventory["sites"]),
        svs=f"{facets['total']:,}",
        svs_cases=f"{facets['cases']:,}",
        svs_linked=f"{facets['with_data']:,}",
        svs_linked_pct=f"{100 * facets['with_data'] / facets['total']:.0f}",
        probe_errors=f"{inventory['probe_errors']:,}",
        sites=bars(inventory["sites"]),
        formats=bars(inventory["formats"]),
        power=bars(inventory["power"]),
        stains=bars(facet_series(facets["stains"])),
        organs=bars(facet_series(facets["organs"])),
        causes_n=f"{causes['n_cases']:,}",
        causes_sites=len(causes["sites"]),
        causes_k=len(causes["diseases"]),
        causes=bars(top, causes["n_cases"]),
        mix=stacked(causes, names),
        organ=settings["organ"],
        organ_slides=f"{organ['count']:,}",
        branch=html.escape(embedding["branch"]),
        projection=settings["projection"].upper(),
        method=settings["cluster_method"].capitalize(),
        n_clusters=n_clusters,
        patients=f"{cohort['included']:,}",
        associations=association_rows(embedding["associations"]),
        observations=observations(inventory, causes, embedding["associations"]),
    )
    return template.substitute(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/eda-report.toml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    out = settings["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    inventory = inventory_summary(settings["inventory"])
    svs = svs_summary(settings["svs_viewer"], settings["top_diseases"])
    embedding = embedding_summary(settings)
    with chromium() as browser:
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        names = shoot_map(page, settings["svs_viewer"], out)
        shoot_graph(page, settings, out)
        report = out / "report.html"
        report.write_text(render(settings, inventory, svs, embedding, names))
        page.goto(report.as_uri())
        page.pdf(path=out / "eda-report.pdf", format="A4", print_background=True)
    print(out / "eda-report.pdf")


if __name__ == "__main__":
    main()
