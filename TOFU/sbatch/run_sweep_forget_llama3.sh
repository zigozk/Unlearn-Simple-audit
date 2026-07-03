#!/bin/bash
#SBATCH -J sweep_llama31_forget
#SBATCH -o logs/sweep_llama31_%j.out
#SBATCH -e logs/sweep_llama31_%j.err
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

PYTHON_SCRIPT="forget.py"

# ===== 2) 你需要改的：Llama3.1 fine-tuned 模型路径 =====
MODEL_PATH="/home/zkzhang/unlearning/Unlearn-Simple/TOFU/paper_models/ft_llama3.1-8b_TOFU_full_lr1e-5_seed42_run1"   # <- 改这里
MODEL_FAMILY="llama3.1-8b"

# ===== 3) 固定参数（按需改）=====
SPLIT="forget05"
BATCH_SIZE=2
ACCUM_STEPS=16
NUM_EPOCHS=5
NPO_COEFF="1.0"
GAMMA="0.0"
WEIGHT_DECAY="0.01"

# ===== 4) Sweep 网格（小范围够用）=====
FORGET_LOSSES=(npo simnpo simnpo_grad_diff npo_grad_diff)
BETAS=(0.05 0.1 0.2)
LAMBDAS=(0.05 0.15 0.3)         # 这里对应 grad_diff_coeff
LRS=(5e-6 1e-5 1e-4)

BASE_SAVE_ROOT="/home/zkzhang/unlearning/Unlearn-Simple/TOFU/unlearned/llama31_sweeps"
for FORGET_LOSS in "${FORGET_LOSSES[@]}"; do
  for LR in "${LRS[@]}"; do
    for BETA in "${BETAS[@]}"; do
      for LAMBDA in "${LAMBDAS[@]}"; do

        SAVE_DIR="${BASE_SAVE_ROOT}/${FORGET_LOSS}_${SPLIT}_lr${LR}_beta${BETA}_lambda${LAMBDA}_bs${BATCH_SIZE}_ga${ACCUM_STEPS}_ep${NUM_EPOCHS}"
        echo "=========================================================="
        echo "RUN: loss=${FORGET_LOSS}, lr=${LR}, beta=${BETA}, lambda=${LAMBDA}"
        echo "SAVE_DIR: ${SAVE_DIR}"
        echo "=========================================================="

        python ${PYTHON_SCRIPT} \
          --config-name forget \
          model_family=${MODEL_FAMILY} \
          model_path=${MODEL_PATH} \
          forget_loss=${FORGET_LOSS} \
          split=${SPLIT} \
          batch_size=${BATCH_SIZE} \
          gradient_accumulation_steps=${ACCUM_STEPS} \
          num_epochs=${NUM_EPOCHS} \
          lr=${LR} \
          beta=${BETA} \
          grad_diff_coeff=${LAMBDA} \
          npo_coeff=${NPO_COEFF} \
          gamma=${GAMMA} \
          weight_decay=${WEIGHT_DECAY} \
          LoRA.r=0 \
          save_dir=${SAVE_DIR} \
          overwrite_dir=true

      done
    done
  done
done

echo "All sweeps finished. Results under: ${BASE_SAVE_ROOT}"
