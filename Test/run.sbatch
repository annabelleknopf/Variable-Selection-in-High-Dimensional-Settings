#!/bin/bash
#SBATCH -J test_cpu
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --container-image=/dss/dsshome1/0C/ra96quq2/dev_clean_fixed.sqsh

set -e
mkdir -p logs

# start in the directory from which you ran "sbatch"
cd "$SLURM_SUBMIT_DIR"

python3 test.py          # <-- works because we’re now in /home/…/MA/Test
# or, if you really want an absolute path:
# python3 "$HOME/MA/Test/test.py"