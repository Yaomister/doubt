#!/bin/bash
#SBATCH --job-name=doubt-poc_2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/doubt-poc_2_%A_%a.out
#SBATCH --array=0-6

module load python/3.13.5
source /home/yao.eric/doubt/.venv/bin/activate
export HF_ALLOW_CODE_EVAL=1

RUNS=(
    "--tasks mbpp --num_fewshot 3 --limit 500 --model_args method=rethink,epsilon=0.0001"
    "--tasks mbpp --num_fewshot 3 --limit 500 --model_args method=rethink,epsilon=0.0002"
    "--tasks mbpp --num_fewshot 3 --limit 500 --model_args method=rethink,epsilon=0.0004"
)

NAME=doubt-poc_$SLURM_ARRAY_TASK_ID
python harness.py --model doubt ${RUNS[$SLURM_ARRAY_TASK_ID]} --output_path results/$NAME --log_samples --confirm_run_unsafe_code