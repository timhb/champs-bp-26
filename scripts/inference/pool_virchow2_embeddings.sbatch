#!/bin/bash
#SBATCH --job-name=virchow2-meanpool
#SBATCH --account=sci-lippert-bp2026cl1
#SBATCH --partition=pot-hpi-gpu-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --chdir=/sc/home/tim.beier/workspace/projects/bp2026cl1
#SBATCH --output=virchow2-meanpool-%j.log

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# The default Pixi environment has CPU-only PyTorch. Use the isolated CUDA build.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi
pixi run --frozen -e pooling-gpu python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True); assert torch.cuda.is_available(), "CUDA-enabled PyTorch cannot access the allocated GPU"; print("GPU:", torch.cuda.get_device_name(), flush=True)'
pixi run --frozen -e pooling-gpu python -m bp26.inference.meanpool --device cuda "$@"
