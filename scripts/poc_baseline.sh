#!/bin/bash
#SBATCH --job-name=doubt-poc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/doubt-smoke_%A_%a.out
#SBATCH --array=0-3

source /home/yao.eric/doubt/.venv/bin/activate


time python experiment.py --smoke-test --method baseline