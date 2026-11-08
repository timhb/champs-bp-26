# champs-bp-26

Exploration and modelling of CHAMPS MITS pathology slides (whole-slide images plus case data).
The dataset lives on the read-only share `/sc/projects/sci-lippert/champs`; nothing from it is
committed.

## Setup

Linux only. Install Pixi ([guide](docs/guides/pixi.md)), then from the repository root:

```bash
pixi install --locked
pixi run playwright install chromium   # headless browser for screenshots and PDFs (once per user)
```

Run CPU work on an interactive node and GPU jobs through Slurm ([guide](docs/guides/slurm.md)).

## Tools

| Command | What it does |
| --- | --- |
| `pixi run svs-viewer` | Slide browser with case data, disease map and data-link diagram on port 8765 |
| `pixi run attention-viewer` | CLAM-MB attention heatmaps, slide graph and cluster associations on port 8766 |
| `pixi run eda-report` | EDA report PDF built from both running viewers (`outputs/eda-report/`) |
| `pixi run inference` | Score Virchow2 bags with the stored lung/liver champion weights |
| `pixi run check` | Lint and format checks |

Both viewers bind to localhost; reach them through port forwarding (e.g. the VS Code Ports panel).

**SVS viewer.** Settings: [svs_viewer.toml](notebooks/svs-viewer/configs/svs_viewer.toml). The
first start scans the share and caches the slide list in `data/processed/svs-viewer/`
(roughly 10–15 min); later starts are instant. Pass `--rescan` after new slides arrive. Case
tables load about 20 s after start.

**Attention viewer.** Settings: [attention-viewer.toml](configs/attention-viewer.toml). It
needs a slide manifest and, for the graph page, a CLAM-MB embedding export:

```bash
pixi run python scripts/inference/build_manifest.py --out data/inference/slides.csv
pixi run python scripts/inference/pool_clam_embeddings.py --manifest data/inference/slides.csv \
    --out outputs/clam-mb --read-workers 16
```

The export is I/O-bound: about 0.3 s per slide with 16 read workers on 4 CPUs (single run),
so ~1 h for the 11k liver slides, ~3 h with lung. The graph shows liver only, so a liver-only
manifest suffices. Use `--limit` with a separate `--out` for a smoke test.

**Headless browser.** `bp26.browser.chromium()` launches Chromium for screenshots, PDFs and UI
checks on the cluster nodes (see [eda_report.py](src/bp26/eda_report.py) for an example).

## Layout

| Path | Contents |
| --- | --- |
| `src/bp26/` | Package code: `attention/` viewer backend, `inference/` CLAM pipeline, EDA report |
| `notebooks/` | Viewer front ends (`svs-viewer/` also holds its Python backend and tests) |
| `configs/` | Run and viewer settings |
| `scripts/` | Entry points and Slurm batch scripts |
| `tests/` | Tests for `src/bp26/` |
| `data/`, `outputs/`, `logs/` | Local, gitignored |

## Docs

- [Rules](docs/rules.md): conventions for code, data and collaboration
- [Experiments](docs/experiments.md): log of reported runs (check it before starting one)
- [Dependencies](docs/dependencies.md): why each package is here
