#!/bin/bash

# DATA_DIR is the inherited by top-level script,
# it's the root directory of the test data
IMAGENET_DIR=$DATA_DIR/imagenet-tiny/tfrecords

TRAIN_EPOCHS=3

NUM_IMAGES=100000,10000
NUM_CLASSES=200

DEFAULT_IMAGE_SIZE=64

RESIZE_MIN=$DEFAULT_IMAGE_SIZE

args=(
    "--tpu=$TPU_NAME"
    "--data_dir=$IMAGENET_DIR"
    "--batch_size=1024"
    "--steps_per_loop=500"
    "--train_epochs=$TRAIN_EPOCHS"
    "--use_synthetic_data=false"
    "--dtype=fp32"
    "--enable_eager=true"
    "--enable_tensorboard=true"
    "--distribution_strategy=tpu"
    "--log_steps=50"
    "--single_l2_loss_op=true"
    "--use_tf_function=true"
    "--num_images=$NUM_IMAGES"
    "--num_classes=$NUM_CLASSES"
    "--default_image_size=$DEFAULT_IMAGE_SIZE"
    "--resize_min=$RESIZE_MIN"
)

echo "Using ${args[@]}"

cd official/legacy/image_classification/resnet
python3 resnet_ctl_imagenet_main.py "${args[@]}" --model_dir=$MODEL_DIR
