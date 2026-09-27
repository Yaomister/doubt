#!/bin/bash
#SBATCH --job-name=doubt-experiment_3_a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/doubt-exp3a_%A_%a.out
#SBATCH --array=0-7

module load python/3.13.5
source /home/yao.eric/doubt/.venv/bin/activate
export HF_ALLOW_CODE_EVAL=1

RUNS=(
  "--tasks gsm8k --num_fewshot 4 --model_args method=baseline,steps=512"
  "--tasks minerva_math --num_fewshot 4 --model_args method=baseline,steps=512"
  "--tasks humaneval --num_fewshot 0 --confirm_run_unsafe_code --model_args method=baseline,steps=512"
  "--tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code --model_args method=baseline,steps=512"
  "--tasks gsm8k --num_fewshot 4 --model_args method=margin,steps=512"
  "--tasks minerva_math --num_fewshot 4 --model_args method=margin,steps=512"
  "--tasks humaneval --num_fewshot 0 --confirm_run_unsafe_code --model_args method=margin,steps=512"
  "--tasks mbpp --num_fewshot 3 --confirm_run_unsafe_code --model_args method=margin,steps=512"
)

NAME=doubt-exp3a_$SLURM_ARRAY_TASK_ID
python eval_doubt.py --model doubt ${RUNS[$SLURM_ARRAY_TASK_ID]} --output_path results/$NAME --log_samples --use_cache cache/${NAME}_