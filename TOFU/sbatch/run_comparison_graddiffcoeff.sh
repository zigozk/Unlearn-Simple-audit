#!/bin/bash
#SBATCH -J com_gdcoeff         
#SBATCH -o logs/com_gdcoeff_%j.out              
#SBATCH -e logs/com_gdcoeff_%j.err              
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
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK
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
BATCH_SIZE=2
ACCUM_STEPS=16
NUM_EPOCHS=10
LR="1e-5"
BETA="2.5"
SPLIT="forget05"
COEFF=0.1375

# ==================== 2. 实验组 A: 正确参数 (SimNPO Paper) ====================
# grad_diff_coeff = 0.15 (论文推荐范围 0.05-0.25)
# 预期：Forget Quality 高，Utility 正常

LAMBDA_A="0.15"
SAVE_DIR_A="/home/zkzhang/unlearning/Unlearn-Simple/TOFU/unlearned/SimNPO_Lambda${LAMBDA_A}_Beta${BETA}_Npo_coeff${COEFF}_LR${LR}_epochs${NUM_EPOCHS}"

echo "=========================================================="
echo "STARTING RUN A: SimNPO (Correct Lambda = ${LAMBDA_A})"
echo "Save Dir: ${SAVE_DIR_A}"
echo "=========================================================="

python ${PYTHON_SCRIPT} \
    --config-name forget \
    model_path=${MODEL_PATH} \
    forget_loss=simnpo_grad_diff \
    split=${SPLIT} \
    batch_size=${BATCH_SIZE} \
    gradient_accumulation_steps=${ACCUM_STEPS} \
    num_epochs=${NUM_EPOCHS} \
    lr=${LR} \
    beta=${BETA} \
    grad_diff_coeff=${LAMBDA_A} \
    npo_coeff=${COEFF} \
    gamma=0.0 \
    LoRA.r=0 \
    save_dir=${SAVE_DIR_A} \
    overwrite_dir=true

# ==================== 3. 实验组 B: 对照组 (Default/High Retain) ====================
# grad_diff_coeff = 1.0 (过大的 Retain 权重)
# 预期：Under-forgetting (遗忘不干净)，Forget Quality 低

LAMBDA_B="1.0"
SAVE_DIR_B="/home/zkzhang/unlearning/Unlearn-Simple/TOFU/unlearned/SimNPO_Lambda${LAMBDA_B}_Beta${BETA}_Npo_coeff${COEFF}_LR${LR}_epochs${NUM_EPOCHS}"

echo "=========================================================="
echo "STARTING RUN B: SimNPO (Control Lambda = ${LAMBDA_B})"
echo "Save Dir: ${SAVE_DIR_B}"
echo "=========================================================="

python ${PYTHON_SCRIPT} \
    --config-name forget \
    model_path=${MODEL_PATH} \
    forget_loss=simnpo_grad_diff \
    split=${SPLIT} \
    batch_size=${BATCH_SIZE} \
    gradient_accumulation_steps=${ACCUM_STEPS} \
    num_epochs=${NUM_EPOCHS} \
    lr=${LR} \
    beta=${BETA} \
    grad_diff_coeff=${LAMBDA_B} \
    npo_coeff=${COEFF} \
    gamma=0.0 \
    LoRA.r=0 \
    save_dir=${SAVE_DIR_B} \
    overwrite_dir=true

echo "=========================================================="
echo "All experiments finished."
echo "Check results in:"
echo "1. ${SAVE_DIR_A}/eval_log_aggregated.json"
echo "2. ${SAVE_DIR_B}/eval_log_aggregated.json"
echo "=========================================================="