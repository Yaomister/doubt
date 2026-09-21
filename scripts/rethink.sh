#!/bin/bash
#SBATCH --job-name=doubt-rethink
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/doubt-rethink_%A_%a.out
#SBATCH --array=0-7

source /home/yao.eric/doubt/.venv/bin/activate

RUNS=(
  "--method rethink --epsilon 0.005 --dataset gsm8k"
  "--method rethink --epsilon 0.005 --dataset math"
  "--method rethink --epsilon 0.005 --dataset humaneval"
  "--method rethink --epsilon 0.005 --dataset mbpp"
  "--method rethink --epsilon 0.005 --random --dataset gsm8k"
  "--method rethink --epsilon 0.005 --random --dataset math"
  "--method rethink --epsilon 0.005 --random --dataset humaneval"
  "--method rethink --epsilon 0.005 --random --dataset mbpp"
)

python experiment.py ${RUNS[$SLURM_ARRAY_TASK_ID]}