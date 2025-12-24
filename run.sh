#!/usr/bin/env bash
set -e

# =========================
# SETTINGS
# =========================

IMAGE_PATH="PATH TO YOUR IMAGE HERE"
SCRIPT_PATH="ai examples/predict_one.py"
CKPT_PATH="ai examples/resnet18_biomes.pth"
IMG_SIZE=224
TOPK=5


python "$SCRIPT_PATH" \
  --image "$IMAGE_PATH" \
  --ckpt "$CKPT_PATH" \
  --img_size "$IMG_SIZE" \
  --topk "$TOPK"