#!/bin/bash
#SBATCH --job-name=mbpp-dream
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mbpp-dream_%A_%a.out
#SBATCH --array=0-5

module load python/3.13.5
source /home/yao.eric/doubt/.venv/bin/activate
export HF_ALLOW_CODE_EVAL=1

RUNS=(
    "--tasks mbpp --num_fewshot 3 --model_args model_type=dream,method=rethink"
    "--tasks mbpp --num_fewshot 3 --model_args model_type=dream,method=klass"
    "--tasks mbpp --num_fewshot 3 --model_args model_type=dream,method=top"
    "--tasks mbpp --num_fewshot 3 --model_args model_type=dream,method=top-margin"
    "--tasks mbpp --num_fewshot 3 --model_args model_type=dream,method=random"
    "--tasks mbpp --num_fewshot 3 --model_args model_type=dream,method=rethink,apply_core=True"
)

NAME=mbpp-dream_$SLURM_ARRAY_TASK_ID
python harness.py --model doubt ${RUNS[$SLURM_ARRAY_TASK_ID]} --output_path results/$NAME --log_samples --confirm_run_unsafe_code