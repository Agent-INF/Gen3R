#!/bin/bash
# Training script for Gen3R
# Based on the paper: Gen3R: 3D Scene Generation Meets Feed-Forward Reconstruction
# Paper: https://arxiv.org/abs/2601.04090
#
# This script runs the two-stage training process:
# Stage 1: Train Geometry Adapter
# Stage 2: Fine-tune Video Diffusion Model for joint generation

set -e

# Configuration
DATA_ROOT="${DATA_ROOT:-./data}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
NUM_GPUS="${NUM_GPUS:-1}"

# Model paths (download from HuggingFace if not present)
VGGT_PATH="${CHECKPOINTS_DIR}/vggt"
WAN_VAE_PATH="${CHECKPOINTS_DIR}/wan_vae"

# Check if checkpoints exist
if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "Downloading pre-trained checkpoints..."
    git lfs install
    git clone https://huggingface.co/JaceyH919/Gen3R "$CHECKPOINTS_DIR"
fi

echo "=========================================="
echo "Gen3R Training Pipeline"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo "Checkpoints: $CHECKPOINTS_DIR"
echo "Output: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="

# Stage 1: Train Geometry Adapter
# ================================
# This stage trains the adapter to map VGGT tokens to a latent space
# aligned with the appearance VAE latent space

echo ""
echo "=========================================="
echo "Stage 1: Training Geometry Adapter"
echo "=========================================="

STAGE1_OUTPUT="${OUTPUT_DIR}/stage1_geometry_adapter"
mkdir -p "$STAGE1_OUTPUT"

# Initial training with 25 frames (shorter sequences)
echo "Stage 1a: Initial training with 25 frames..."
python train_adapter.py \
    --data_root "$DATA_ROOT" \
    --vggt_path "$VGGT_PATH" \
    --wan_vae_path "$WAN_VAE_PATH" \
    --output_dir "${STAGE1_OUTPUT}/initial" \
    --num_epochs 50 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --num_frames 25 \
    --resolution 560 \
    --lambda_rec 1.0 \
    --lambda_kl 0.001 \
    --save_every 1000 \
    --mixed_precision bf16 \
    --seed 42

# Fine-tuning with 49 frames (longer sequences)
echo "Stage 1b: Fine-tuning with 49 frames..."
LATEST_CHECKPOINT=$(find "${STAGE1_OUTPUT}/initial" -name "final_checkpoint.pt" 2>/dev/null | head -1)

# Check if checkpoint exists
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Warning: No checkpoint found from Stage 1a, looking for most recent checkpoint..."
    LATEST_CHECKPOINT=$(find "${STAGE1_OUTPUT}/initial" -name "checkpoint-*.pt" 2>/dev/null | sort -V | tail -1)
fi

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Warning: No checkpoint found, starting Stage 1b from scratch"
    CHECKPOINT_ARG=""
else
    echo "Using checkpoint: $LATEST_CHECKPOINT"
    CHECKPOINT_ARG="--checkpoint_path $LATEST_CHECKPOINT"
fi

python train_adapter.py \
    --data_root "$DATA_ROOT" \
    --vggt_path "$VGGT_PATH" \
    --wan_vae_path "$WAN_VAE_PATH" \
    --output_dir "${STAGE1_OUTPUT}/finetune" \
    $CHECKPOINT_ARG \
    --num_epochs 20 \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --num_frames 49 \
    --resolution 560 \
    --lambda_rec 1.0 \
    --lambda_kl 0.001 \
    --save_every 500 \
    --mixed_precision bf16 \
    --seed 42

echo "Stage 1 completed!"

# Stage 2: Fine-tune Video Diffusion Model
# ==========================================
# This stage fine-tunes the diffusion model to jointly generate
# appearance and geometry latent codes

echo ""
echo "=========================================="
echo "Stage 2: Fine-tuning Video Diffusion Model"
echo "=========================================="

STAGE2_OUTPUT="${OUTPUT_DIR}/stage2_diffusion"
mkdir -p "$STAGE2_OUTPUT"

# Copy trained geometry adapter to checkpoints
TRAINED_ADAPTER=$(find "${STAGE1_OUTPUT}/finetune" -name "geo_adapter-*" -type d 2>/dev/null | sort | tail -1)
if [ -n "$TRAINED_ADAPTER" ] && [ -d "$TRAINED_ADAPTER" ]; then
    mkdir -p "${CHECKPOINTS_DIR}/geo_adapter/"
    cp -r "$TRAINED_ADAPTER"/* "${CHECKPOINTS_DIR}/geo_adapter/"
    echo "Updated geometry adapter in checkpoints from: $TRAINED_ADAPTER"
else
    echo "Warning: No trained geometry adapter found, using existing checkpoint"
fi

echo "Training diffusion model..."
python train_diffusion.py \
    --data_root "$DATA_ROOT" \
    --pretrained_path "$CHECKPOINTS_DIR" \
    --output_dir "$STAGE2_OUTPUT" \
    --num_epochs 50 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --num_frames 49 \
    --resolution 560 \
    --text_dropout_prob 0.1 \
    --camera_dropout_prob 0.1 \
    --task_1view_prob 0.4 \
    --task_2view_prob 0.3 \
    --task_allview_prob 0.3 \
    --save_every 500 \
    --mixed_precision bf16 \
    --seed 42

echo "Stage 2 completed!"

echo ""
echo "=========================================="
echo "Training Pipeline Completed!"
echo "=========================================="
echo "Final outputs saved to: $OUTPUT_DIR"
echo ""
echo "To use the trained model, update the checkpoint path in inference:"
echo "  python infer.py --pretrained_model_name_or_path ${STAGE2_OUTPUT}/final_pipeline ..."
