# Slurm

Always pass the project account `-A sci-lippert-bp2026cl1`.

| Partition | Use | Limit | GPU types (`--gres=gpu:TYPE:N`) |
| --- | --- | --- | --- |
| `pot-hpi-cpu-interactive` (default) | CPU shell, setup, checks | 8 h | — |
| `pot-hpi-cpu-batch` | CPU batch jobs | 7 d | — |
| `pot-hpi-gpu-interactive` | GPU shell, debugging | 8 h | `a100`, `a40`, `gh200`, `2080_ti` |
| `pot-hpi-gpu-shortrun` | Quick GPU tests | 1 h | `a100`, `b200`, `rtx_pro_6000` |
| `pot-hpi-gpu-batch` | Training | 7 d | `a100`, `b200`, `rtx_pro_6000`, `gh200` |

Check live availability with `sinfo -s` or `sinfo -p PARTITION -o "%n %G %c %m %t"`.

## Interactive node

```bash
srun -A sci-lippert-bp2026cl1 -p pot-hpi-gpu-interactive \
     --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
```

Default memory is only 1.5 GB per CPU, so set `--mem`. Shorter `--time` starts sooner. Exit
the shell to release the node.

## Batch job

```bash
#!/bin/bash
#SBATCH -A sci-lippert-bp2026cl1
#SBATCH -p pot-hpi-gpu-batch
#SBATCH --job-name=NAME
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out   # %x job name, %j job id

pixi run --frozen python -m bp26.MODULE
```

Submit from the repository root with `mkdir -p logs && sbatch SCRIPT`. The default environment
has CPU-only PyTorch; GPU code runs in the `pooling-gpu` environment (`pixi run -e pooling-gpu`).

| Command | Purpose |
| --- | --- |
| `squeue --me` | Your jobs (PD pending, R running) |
| `scontrol show job ID` | Details, including why a job is pending |
| `sacct -j ID --format=JobID,State,Elapsed,MaxRSS` | Finished job stats |
| `srun --jobid=ID --overlap nvidia-smi` | GPU use of a running job |
| `scancel ID` | Cancel |
