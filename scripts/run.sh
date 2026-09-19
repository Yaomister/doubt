#!/bin/bash
#SBATCH --job-name=doubt
#SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1       
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%a_%A.out

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