#!/bin/bash
# LoRA fine-tune OpenVLA on LIBERO-Spatial (run this on GPU-mode instance)
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla
cd /root/autodl-tmp/openvla
export PYTHONPATH=/root/autodl-tmp/openvla
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /root/autodl-tmp/exp/libero_spatial_lora

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path /root/autodl-tmp/checkpoints/openvla-7b \
  --data_root_dir /root/autodl-tmp/datasets/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /root/autodl-tmp/exp/libero_spatial_lora \
  --adapter_tmp_dir /root/autodl-tmp/exp/libero_spatial_lora/adapters \
  --lora_rank 32 \
  --batch_size 2 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --image_aug True \
  --max_steps 10000 \
  --save_steps 1000 \
  --wandb_project openvla-libero-repro \
  --wandb_entity none
