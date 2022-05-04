#!/bin/bash

cd official/legacy/image_classification

python3 classifier_trainer.py \
        --mode=train_and_eval \
        --model_type=efficientnet \
        --dataset=imagenet \
        --tpu=$TPU_NAME \
        --model_dir=$MODEL_DIR \
        --data_dir=$DATA_DIR/imagenet2012/tfrecords \
        --config_file=configs/efficientnet-b0-tpu.yaml
