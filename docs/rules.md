# Rules

Current conventions only. Edit in place; history lives in Git.

## Environment

- Pixi manages dependencies and tasks ([guide](guides/pixi.md)); Python is pinned to 3.14.
- Add dependencies with `pixi add`, never by hand; list them in [dependencies.md](dependencies.md).
- GPU work runs through Slurm ([guide](guides/slurm.md)).
- Browser screenshots, PDFs and UI checks use `bp26.browser.chromium()` (once: `pixi run playwright install chromium`).

## Repository

- Code: `src/bp26/`. Experiment settings: `configs/`. Exploration: `notebooks/`.
- `data/`, `outputs/`, `checkpoints/` and `logs/` are gitignored; never commit data.
- The CHAMPS dataset share is read-only.

## Code quality

- Ruff (settings in `pyproject.toml`); `pixi run check` must pass before merging.

## Collaboration

- Work on branches, merge through PRs with the template in `.github/`.
- Only squash-merge into `main`. Title: `Type/message` (`Feat`, `Fix`, `Docs`, `Chore`, `Exp`),
  where the message completes "If merged, this change will …", e.g. `Feat/add Slurm guide`.
  Special commits (e.g. reverts) may drop `Type/`.
- Log every reported run in [experiments.md](experiments.md).
