#!/bin/bash

TRAIN_FILE_PATTERN="${DATA_DIR?}/coco/train-*-of-*.tfrecord"
EVAL_FILE_PATTERN="${DATA_DIR?}/coco/val-*-of-*.tfrecord"
VAL_JSON_FILE="${DATA_DIR?}/coco/raw-data/annotations/instances_val2017.json"

PARAMS="
type: retinanet
architecture:
  backbone: spinenet
  multilevel_features: identity
spinenet:
  model_id: 49
train:
  total_steps: 1000
  train_file_pattern: ${TRAIN_FILE_PATTERN?}
eval:
  val_json_file: ${VAL_JSON_FILE?}
  eval_file_pattern: ${EVAL_FILE_PATTERN?}
"

cd ~/models/official/legacy/detection

for i in {1..5}
do
  python3 main.py \
    --strategy_type=tpu \
    --tpu="${TPU_NAME?}" \
    --model_dir="${MODEL_DIR?}/run$i" \
    --mode=train \
    --params_override="${PARAMS?}" \
    "$@"
done