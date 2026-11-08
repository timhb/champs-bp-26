# champs-bp-26

## Project structure

Path | Purpose |
| --- | --- |
| [configs/](configs/) | Experiment settings, model parameters, and training configurations. |
| [data/](data/) | Local datasets and derived data. |
| [docs/](docs/) | Research plans, experiment records, project decisions, and practical guides. |
| [notebooks/](notebooks/) | Exploratory analysis and interactive visualizations. |
| [outputs/](outputs/) | Generated results, figures, logs, and model artifacts. |
| [scripts/](scripts/) | Commands and helper scripts for running project workflows. |
| [src/](src/) | Reusable Python source code. |
| [tests/](tests/) | Automated checks of code behavior and research-critical assumptions. |


## Environment and development

We use **Pixi** ✨ to manage dependencies and run project commands.

### Setup

This project targets **Linux**. If you have not installed or used Pixi before,
start with the [Pixi guide for Linux](docs/guides/pixi.md). It covers installation,
environment setup, daily commands, dependency changes, and cluster cache warnings.

### Dependency changes

- Document each dependency's purpose in [docs/dependencies.md](docs/dependencies.md).
- Keep conventions current in [docs/rules.md](docs/rules.md).

## Training environment

Our usual training hardware is **NVIDIA A100**.

Use **CPU** nodes for environment setup, coding and CPU-compatible tasks.
Run **GPU** training on an allocated A100 node using the cluster's job workflow. For larger jobs, consider running these as a batch job.

To avoid duplicate runs, check [docs/experiments.md](docs/experiments.md) before
starting an experiment and the run you perform. Include your name,
the configuration used, results (e.g. AUROC, F1, accuracy), GPU model/count,
and runtime.

## Dataset

We use **CHAMPS MITS pathology samples**. Keep dataset files outside Git and
document access/setup instructions without including credentials.

## SVS viewer

A read-only browser for the CHAMPS slides and their case data (lab, TAC, cause of death,
CPL reports), with a disease map and a diagram of how the tables link. It lives in
[notebooks/svs-viewer/](notebooks/svs-viewer/) and never writes to the dataset share.

```bash
pixi run viewer            # serves http://127.0.0.1:8765
pixi run viewer --rescan   # re-scan the slide folder after new slides arrive
```

Run it on a compute node and forward port 8765 (e.g. via VS Code's Ports panel). Paths and
the port are set in [notebooks/svs-viewer/configs/svs_viewer.toml](notebooks/svs-viewer/configs/svs_viewer.toml). The first start scans
the slide folder and caches the list in `data/processed/svs-viewer/` (gitignored); case
tables take about 20 seconds to load.
