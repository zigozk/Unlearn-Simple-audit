#!/bin/bash
#SBATCH -J eval
#SBATCH -o logs/eval_llama31_%j.out
#SBATCH -e logs/eval_llama31_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a100-sxm4-80gb:1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + RANDOM % 10000))
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export HYDRA_FULL_ERROR=1
export WANDB_DISABLED=true
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK
unset SLURM_PROCID SLURM_LOCALID SLURM_NODEID

set -e  # 遇到错误立即停止

# ==================== 0. 环境准备 ====================
. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/12.1

. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate tofu_llama3

cd $HOME/unlearning/Unlearn-Simple/TOFU

BASE_ROOT=

python evaluate_util.py \
  --config-path config \
  --config-name eval_everything \
  use_pretrained=false \
  model_family=llama3.1-8b \
  model_path="/home/zkzhang/unlearning/Unlearn-Simple/TOFU/unlearned/llama31_forget/npo_forget05_lr1e-5_beta0.2_lambda0.15_bs2_ga16_ep5" \
  save_dir="/home/zkzhang/unlearning/Unlearn-Simple/TOFU/unlearned/llama31_forget/npo_forget05_lr1e-5_beta0.2_lambda0.15_bs2_ga16_ep5/eval" \
  overwrite=true \
  batch_size=2