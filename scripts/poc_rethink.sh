#!/bin/bash
#SBATCH --job-name=doubt-poc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/doubt-smoke_%A_%a.out

module load python/3.13.5
source /home/yao.eric/doubt/.venv/bin/activate

python experiment.py --smoke-test --method rethink --epsilon 0.005