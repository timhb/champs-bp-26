#!/bin/bash
#SBATCH --job-name=virchow2-meanpool-cpu
#SBATCH --account=sci-lippert-bp2026cl1
#SBATCH --partition=pot-hpi-cpu-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --chdir=/sc/home/tim.beier/workspace/projects/bp2026cl1
#SBATCH --output=virchow2-meanpool-cpu-%j.log

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

pixi run --frozen python -m bp26.inference.meanpool --device cpu "$@"
