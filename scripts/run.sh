#!/bin/bash
#SBATCH --job-name=doubt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1       
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%a_%A.out

set -eu
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate doubt   # EDIT
cd "${SLURM_SUBMIT_DIR:-.}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

EPS=${EPS:-0.05}
RUNS=(
  "--method baseline"
  "--method rethink  --epsilon $EPS"
  "--method rethink  --epsilon $EPS --random"
  "--method pressure --epsilon $EPS"
  "--method compare  --epsilon $EPS"
)

echo "=== ${RUNS[$SLURM_ARRAY_TASK_ID]} on $(hostname)"
python doubt.py ${RUNS[$SLURM_ARRAY_TASK_ID]}