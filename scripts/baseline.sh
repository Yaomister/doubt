#!/bin/bash
#SBATCH --job-name=doubt-main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/doubt-main_%A_%a.out
#SBATCH --array=0-7

source /home/yao.eric/doubt/.venv/bin/activate

RUNS=(
  "--method baseline --dataset gsm8k"
  "--method baseline --dataset math"
  "--method baseline --dataset humaneval"
  "--method baseline --dataset mbpp"
  "--method core     --dataset gsm8k"
  "--method core     --dataset math"
  "--method core     --dataset humaneval"
  "--method core     --dataset mbpp"
)

python experiment.py ${RUNS[$SLURM_ARRAY_TASK_ID]}