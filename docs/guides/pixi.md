# Using Pixi on Linux

This guide is for **Linux only**, including our cluster's CPU and NVIDIA A100
nodes. Our project currently targets `linux-64` (x86-64 Linux).

Pixi installs the project's dependencies into a local environment and runs
commands inside it. You do not need to install Conda or Python separately to
get started with Pixi.

## 1. Install Pixi

Connect to the cluster first. Run this command inside the cluster to install pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

The installer downloads Pixi to `~/.pixi/bin` and updates your shell's startup
configuration.  
 Open a new terminal or reconnect
your SSH session, then verify the installation:

```bash
pixi --version
```

If you see `pixi: command not found`, enable it in your current terminal:

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
```

See the [official installation guide](https://pixi.prefix.dev/latest/installation/)
for installer details.

## 2. Install all dependencies

Go into the root directory of the repository. Adjust the path if it is elsewhere:

```bash
cd champs-bp-26
pixi install --locked
```


| File or directory | Role |
| --- | --- |
| `pyproject.toml` | Package metadata, allowed dependencies, task commands, and Ruff configuration. |
| `pixi.lock` | Exact resolved dependency versions; commit this with dependency changes. |
| `.pixi/` | Generated project environments and local settings; excluded from Git. |

`--locked` checks that the manifest and lockfile agree and installs the
recorded packages without updating the lockfile. This keeps teammates and CI
on the reviewed dependency versions. See the [installation options](https://pixi.prefix.dev/latest/reference/cli/pixi/install/).

## 3. Run project tasks

Run these commands from the repository root. List available tasks with:

```bash
pixi task list
```

| Command | Purpose |
| --- | --- |
| `pixi run lint` | Check Python code for lint violations. |
| `pixi run lint-fix` | Apply safe automatic lint fixes. |
| `pixi run format` | Format Python code. |
| `pixi run format-check` | Check formatting without changes. |
| `pixi run check` | Run lint and formatting checks. |

Tasks are defined under `[tool.pixi.tasks]` in `pyproject.toml`.
`pixi run` selects the project environment automatically; manual activation
is not required. You can also run individual commands inside it:

```bash
pixi run python --version
```

Use `pixi run --locked TASK_NAME` when the lockfile must remain unchanged,
such as in CI. See the [task execution reference](https://pixi.prefix.dev/latest/reference/cli/pixi/run/).

## 4. Everyday workflow

After pulling changes or switching branches, synchronize the environment:

```bash
pixi install --locked
```

After editing Python code:

```bash
pixi run --locked check
```

Inspect automatic changes before committing. If lint violations remain,
fix them manually and repeat the checks. Run relevant tests separately when
changing behavior. The `check` task currently checks lint and formatting only.

## 5. Adding dependencies

Replace `PACKAGE` with the package name. Choose the appropriate source:

| Command | Purpose |
| --- | --- |
| `pixi add PACKAGE` | Add a package from the configured Conda channels. |
| `pixi add --pypi PACKAGE` | Add a package from PyPI. |

These commands update the dependency declarations and lockfile. Prefer them
over installing packages manually inside the environment. Update
`docs/dependencies.md` with the package's purpose.
See the [dependency command reference](https://pixi.prefix.dev/latest/reference/cli/pixi/add/).

## 6. Using Pixi on the cluster

CPU nodes can run setup and code-quality checks. Our usual training hardware
is an **NVIDIA A100 GPU**. Obtain a GPU allocation through the cluster's job
workflow before starting GPU training; `pixi run` does not allocate a GPU.

If Pixi warns that a cache on network storage was redirected to `/tmp`, the
warning itself is not a failed operation. To explicitly use node-local paths
for the metadata caches, run:

```bash
pixi config set --global cache.repodata "/tmp/pixi-cache-$USER/repodata"
pixi config set --global cache.pypi-mapping "/tmp/pixi-cache-$USER/conda-pypi-mapping"
```

See [Pixi cache configuration](https://pixi.prefix.dev/latest/reference/pixi_configuration/#cache).

## 7. Troubleshoot setup

| Message | What to do |
| --- | --- |
| `pixi: command not found` | Reopen the terminal or use the PATH command in step 1. |
| Lockfile missing or out of date | Check that both configuration and lockfile changes were committed. For an intentional dependency change, regenerate with `pixi lock`, inspect the changes, and commit them. Do not delete the lockfile to bypass the check. |
| Task not found | Check the current directory and run `pixi task list`. |
| Package metadata name does not match | Check that `[project].name` matches the local package key under `[tool.pixi.pypi-dependencies]`; also check any existing `setup.py` or `setup.cfg`. |
