#!/bin/bash
suffix="${1?}"

DATA_DIR="$HOME/training-data"
MODEL_DIR="$HOME/data-retinanet-$suffix"

TRAIN_FILE_PATTERN="${DATA_DIR?}/coco/train-*-of-*.tfrecord"
EVAL_FILE_PATTERN="${DATA_DIR?}/coco/val-*-of-*.tfrecord"
VAL_JSON_FILE="${DATA_DIR?}/coco/raw-data/annotations/instances_val2017.json"

PARAMS="
type: retinanet
architecture:
  backbone: spinenet
  multilevel_features: identity
spinenet:
  model_id: '49'
train:
  train_file_pattern: ${TRAIN_FILE_PATTERN?}
eval:
  eval_samples: 5000
  val_json_file: ${VAL_JSON_FILE?}
  eval_file_pattern: ${EVAL_FILE_PATTERN?}
"

# Clear existing directory
rm -rf "${MODEL_DIR?}"

for i in {1..5}
do
  run_dir="${MODEL_DIR?}/run$i"
  python3 main.py \
    --strategy_type=tpu \
    --tpu="local" \
    --model_dir="$run_dir" \
    --mode=train \
    --params_override="${PARAMS?}"
  python3 main.py \
    --strategy_type=tpu \
    --tpu="local" \
    --model_dir="$run_dir" \
    --checkpoint_path="$run_dir" \
    --mode=eval_once \
    --params_override="${PARAMS?}"
  # Clear checkpoints
  echo "Removing checkpoints"
  python3 clear_checkpoints.py "$run_dir"
done
