#!/bin/bash
python3 classifier_trainer.py \
  --mode=train_and_eval \
  --model_type=efficientnet \
  --dataset=imagenet \
  --tpu=local \
  --model_dir="${MODEL_DIR?}" \
  --data_dir="$HOME/training-data/imagenet2012/tfrecords" \
  --config_file=configs/examples/efficientnet/imagenet/efficientnet-b0-tpu.yaml
