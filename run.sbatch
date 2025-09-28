#!/bin/bash
#SBATCH -J simu2_ff
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --container-image=/dss/dsshome1/0C/ra96quq2/dev_clean_fixed.sqsh

set -e
mkdir -p logs

# start in the directory from which you ran "sbatch"
cd "$SLURM_SUBMIT_DIR"

python3 -u simulation2_stepwiseff.py  #Simulation2_50000.py   # <-- works because we’re now in /home/…/MA/
# or, if you really want an absolute path:
# python3 -u "$HOME/MA/Test/test.py"