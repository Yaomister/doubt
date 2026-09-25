#!/bin/bash
#SBATCH --job-name=doubt-matched
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/doubt-matched_%A_%a.out
#SBATCH --array=0-7


module load python/3.13.5
source /home/yao.eric/doubt/.venv/bin/activate

RUNS=(
  "--method baseline --steps 256 --dataset gsm8k"
  "--method baseline --steps 256 --dataset math"
  "--method baseline --steps 256 --dataset humaneval"
  "--method baseline --steps 256 --dataset mbpp"
  "--method core     --steps 256 --dataset gsm8k"
  "--method core     --steps 256 --dataset math"
  "--method core     --steps 256 --dataset humaneval"
  "--method core     --steps 256 --dataset mbpp"
)

python experiment.py ${RUNS[$SLURM_ARRAY_TASK_ID]}