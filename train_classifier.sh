#!/usr/bin/env bash
set -e
DATA_DIR="data/dataset"
MODEL="yolov8n-cls.pt"
EXP_NAME="car_model_exp1"
DEVICE="mps"   # fallback: cpu
IMG=224
EPOCHS=30
BATCH=32

yolo classify train \
  data=${DATA_DIR} \
  model=${MODEL} \
  imgsz=${IMG} \
  epochs=${EPOCHS} \
  batch=${BATCH} \
  lr0=0.01 \
  patience=10 \
  device=${DEVICE} \
  project=runs/classify \
  name=${EXP_NAME}
  plots=False
  