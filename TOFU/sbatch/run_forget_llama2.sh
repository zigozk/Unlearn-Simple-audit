#!/bin/bash
#SBATCH -J npo_fg        
#SBATCH -o logs/npo_fg_%j.out              
#SBATCH -e logs/npo_fg_%j.err              
#SBATCH -p compute                     
#SBATCH -N 1                           
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 6:00:00                    
#SBATCH --gres=gpu:a100-sxm4-80gb:1            

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + RANDOM % 10000))
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export HYDRA_FULL_ERROR=1
export WANDB_DISABLED=true
# unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK
unset SLURM_PROCID SLURM_LOCALID SLURM_NODEID

set -e  # 遇到错误立即停止

# ==================== 0. 环境准备 ====================
. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/11.8


. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate tofucu118 

cd $HOME/unlearning/Unlearn-Simple/TOFU
# ==================== 1. 全局变量定义 ====================
PYTHON_SCRIPT="forget.py" 

MODEL_PATH="/home/zkzhang/unlearning/tofu_ft_llama2-7b"

# 基础参数 (High VRAM Setup: BS=8, Accum=4 => Effective BS=32)
BATCH_SIZE=4
ACCUM_STEPS=8
NUM_EPOCHS=20
LR="1e-5"
BETA="0.2"
SPLIT="forget05"
COEFF=1.0
LAMBDA="0.15"
# ==================== 2. 实验 ====================


SAVE_DIR_A="/home/zkzhang/unlearning/Unlearn-Simple/TOFU/unlearned/NPO_${SPLIT}_Beta${BETA}_Npo_coeff${COEFF}_LR${LR}_epochs${NUM_EPOCHS}"

echo "=========================================================="
echo "STARTING RUN A: SimNPO (Correct Lambda = ${LAMBDA})"
echo "Save Dir: ${SAVE_DIR_A}"
echo "=========================================================="

python ${PYTHON_SCRIPT} \
    --config-name forget \
    model_path=${MODEL_PATH} \
    forget_loss=npo \
    split=${SPLIT} \
    batch_size=${BATCH_SIZE} \
    gradient_accumulation_steps=${ACCUM_STEPS} \
    num_epochs=${NUM_EPOCHS} \
    lr=${LR} \
    beta=${BETA} \
    grad_diff_coeff=${LAMBDA} \
    npo_coeff=${COEFF} \
    gamma=0.0 \
    LoRA.r=0 \
    save_dir=${SAVE_DIR_A} \
    overwrite_dir=true

echo "=========================================================="
echo "All experiments finished."
echo "Check results in:"
echo "${SAVE_DIR_A}/aggregated_stat.txt"
echo "=========================================================="