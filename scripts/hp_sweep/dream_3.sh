#!/bin/bash
#SBATCH --job-name=hp-dream-3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hp-dream-3_%A_%a.out
#SBATCH --array=0-7

module load python/3.13.5
source /home/yao.eric/doubt/.venv/bin/activate
export HF_ALLOW_CODE_EVAL=1

RUNS=(
    "--tasks minerva_math --num_fewshot 4 --limit 200 --model_args model_type=dream,method=rethink,decay=0.95,threshold=2"
    "--tasks minerva_math --num_fewshot 4 --limit 200 --model_args model_type=dream,method=rethink,decay=0.95,threshold=3"
    "--tasks minerva_math --num_fewshot 4 --limit 200 --model_args model_type=dream,method=rethink,decay=0.95,threshold=5"
    "--tasks minerva_math --num_fewshot 4 --limit 200 --model_args model_type=dream,method=rethink,decay=0.95,threshold=8"
    "--tasks humaneval --num_fewshot 0 --limit 200 --model_args model_type=dream,method=rethink,decay=0.9,threshold=1"
    "--tasks humaneval --num_fewshot 0 --limit 200 --model_args model_type=dream,method=rethink,decay=0.9,threshold=2"
    "--tasks humaneval --num_fewshot 0 --limit 200 --model_args model_type=dream,method=rethink,decay=0.9,threshold=3"
    "--tasks humaneval --num_fewshot 0 --limit 200 --model_args model_type=dream,method=rethink,decay=0.9,threshold=5"
)

NAME=hp-dream-3_$SLURM_ARRAY_TASK_ID
python harness.py --model doubt ${RUNS[$SLURM_ARRAY_TASK_ID]} --output_path results/$NAME --log_samples --confirm_run_unsafe_code