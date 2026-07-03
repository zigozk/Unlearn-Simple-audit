#!/bin/bash
#SBATCH -J ft_llama31_tofu   
#SBATCH -o logs/ft_llama31_%j.out              
#SBATCH -e logs/ft_llama31_%j.err              
#SBATCH -p compute                     
#SBATCH -N 1                           
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 24:00:00                    
#SBATCH -w gpu19 --gres=gpu:a100-sxm4-80gb:4       

set -e  # 遇到错误立即停止
. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/12.1

. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate tofu_llama3

cd /home/zkzhang/unlearning/Unlearn-Simple/TOFU


export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/zkzhang/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

# 多机/多卡常用（单机也无害）
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs

# ====== 3) 分布式启动 ======
GPUS=${SLURM_GPUS_PER_NODE:-4}

torchrun --nproc_per_node=${GPUS} \
  finetune.py \
  model_family=llama3.1-8b \
  model_path=/home/share/models/Meta-Llama-3.1-8B-Instruct \
  split=full \
  batch_size=1 \
  gradient_accumulation_steps=32 \
  num_epochs=5 \
  lr=1e-5 \
  weight_decay=0.01 \
  save_dir=paper_models/ft_llama3.1-8b_TOFU_full_lr1e-5_seed42_run1 \
  seed=42 \
  run_index=1
