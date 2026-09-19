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
source /home/yao.eric/doubt/.venv/bin/activate

RUNS=(
  "--method baseline"
  "--method rethink  --epsilon 0.05"
  "--method rethink  --epsilon 0.05 --random"
  "--method pressure --epsilon 0.05"
  "--method compare  --epsilon 0.05"
)

python doubt.py ${RUNS[$SLURM_ARRAY_TASK_ID]}