#!/bin/bash

# DATA_DIR is the inherited by top-level script,
# it's the root directory of the test data
cd official/legacy/image_classification

for i in {1..5}
do
    python3 classifier_trainer.py \
            --mode=train_and_eval \
            --model_type=resnet \
            --dataset=imagenet \
            --tpu=$TPU_NAME \
            --model_dir="$MODEL_DIR/run$i" \
            --data_dir=$DATA_DIR/imagenet2012/tfrecords \
            --config_file=configs/tpu.yaml \
            --seed=1 \
            --enable_op_determinism
done