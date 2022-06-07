#!/bin/bash

cd official/legacy/image_classification

for i in {1..5}
do
        python3 classifier_trainer.py \
                --mode=train_and_eval \
                --model_type=resnet \
                --dataset=imagenet \
                --tpu=$TPU_NAME \
                --model_dir="$MODEL_DIR/run$i" \
                --data_dir=$DATA_DIR/cifar10/3.0.2/ \
                --config_file=configs/tpu_cifar.yaml \
                "$@"
done