#!/bin/bash

cd official/legacy/image_classification

python3 classifier_trainer.py \
        --mode=train_and_eval \
        --model_type=efficientnet \
        --dataset=imagenet \
        --tpu=$TPU_NAME \
        --model_dir=$MODEL_DIR \
        --data_dir=$DATA_DIR/cifar10/3.0.2/ \
        --config_file=configs/efficientnet-b0-tpu_cifar.yaml \
        "$@"
