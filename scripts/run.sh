#!/bin/bash
#SBATCH --job-name=doubt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mu_%A_%a.out
#SBATCH --array=0-4

mkdir -p logs

source /home/yao.eric/doubt/.venv/bin/activate

RUNS=(
  "--method baseline"
  "--method rethink  --epsilon 0.005"
  "--method rethink  --epsilon 0.005 --random"
  "--method pressure --epsilon 0.005"
  "--method compare  --epsilon 0.005"
)

python experiment.py ${RUNS[$SLURM_ARRAY_TASK_ID]}