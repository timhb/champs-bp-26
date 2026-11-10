# Pixi

Pixi installs the project's dependencies into `.pixi/` and runs commands inside it. No separate
Conda or Python install is needed. The project targets `linux-64`.

## Install

On the cluster:

```bash
curl -fsSL https://pixi.sh/install.sh | sh   # installs to ~/.pixi/bin
pixi --version                               # new shell; else: export PATH="$HOME/.pixi/bin:$PATH"
```

Then, in the repository root:

```bash
pixi install --locked   # exact versions from pixi.lock; fails if pyproject.toml and the lock disagree
```

## Daily use

| Command | Purpose |
| --- | --- |
| `pixi install --locked` | Sync the environment after pulling or switching branches |
| `pixi task list` | List project tasks (defined under `[tool.pixi.tasks]` in `pyproject.toml`) |
| `pixi run TASK` | Run a task, e.g. `check`, `lint-fix`, `format`, `svs-viewer` |
| `pixi run python ...` | Run any command inside the environment |

`pixi run` does not allocate a GPU; request one through [Slurm](slurm.md).

## Adding dependencies

```bash
pixi add PACKAGE          # conda-forge (preferred)
pixi add --pypi PACKAGE   # PyPI
```

Commit `pyproject.toml` and `pixi.lock` together and add the package to
[dependencies.md](../dependencies.md). Never install packages by hand inside the environment.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `pixi: command not found` | Open a new shell or export the PATH above |
| Lockfile out of date | Commit both files; for intended changes run `pixi lock`, review, commit. Never delete the lock |
| "cache … redirected to /tmp" warning | Harmless. To silence: `pixi config set --global cache.repodata "/tmp/pixi-cache-$USER/repodata"` (same for `cache.pypi-mapping`) |
| Task not found | Run from the repository root; check `pixi task list` |
