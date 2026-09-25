#!/bin/bash
#SBATCH --job-name=doubt-eps-b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/doubt-eps-b_%A_%a.out
#SBATCH --array=0-3


module load python/3.13.5
source /home/yao.eric/doubt/.venv/bin/activate

RUNS=(
  "--method rethink --epsilon 0.04 --dataset gsm8k"
  "--method rethink --epsilon 0.04 --dataset math"
  "--method rethink --epsilon 0.04 --dataset humaneval"
  "--method rethink --epsilon 0.04 --dataset mbpp"
)

python experiment.py ${RUNS[$SLURM_ARRAY_TASK_ID]}