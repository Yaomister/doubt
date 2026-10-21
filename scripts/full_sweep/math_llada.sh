#!/bin/bash
#SBATCH --job-name=math-llada
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/math-llada_%A_%a.out
#SBATCH --array=0-5

module load python/3.13.5
source /home/yao.eric/doubt/.venv/bin/activate
export HF_ALLOW_CODE_EVAL=1

RUNS=(
    "--tasks minerva_math --num_fewshot 4 --model_args model_type=llada,method=rethink"
    "--tasks minerva_math --num_fewshot 4 --model_args model_type=llada,method=klass"
    "--tasks minerva_math --num_fewshot 4 --model_args model_type=llada,method=top"
    "--tasks minerva_math --num_fewshot 4 --model_args model_type=llada,method=top-margin"
    "--tasks minerva_math --num_fewshot 4 --model_args model_type=llada,method=random"
    "--tasks minerva_math --num_fewshot 4 --model_args model_type=llada,method=rethink,apply_core=True"
)

NAME=math-llada_$SLURM_ARRAY_TASK_ID
python harness.py --model doubt ${RUNS[$SLURM_ARRAY_TASK_ID]} --output_path results/$NAME --log_samples --confirm_run_unsafe_code