#!/bin/bash

# DATA_DIR is the inherited by top-level script,
# it's the root directory of the test data
IMAGENET_DIR=$DATA_DIR/imagenet2012/tfrecords
TRAIN_EPOCHS=90
NUM_IMAGES=1281167,50000

for arg in "$@"
do
    if [ "$arg" == "--test" ]
    then
        echo "Using Tiny ImageNet, and reducing number of epochs."
        IMAGENET_DIR=$DATA_DIR/imagenet-tiny/tfrecords
        TRAIN_EPOCHS=20
        NUM_IMAGES=100000,10000
    fi
done

args=(
    "--tpu=$TPU_NAME"
    "--model_dir=$MODEL_DIR"
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
)

args=$(IFS=' ' echo ${args[*]})

cd official/legacy/image_classification/resnet

for i in {1..5}
do
    python3 resnet_ctl_imagenet_main.py "$args" --deterministic
done

for i in {1..5}
do
    python3 resnet_ctl_imagenet_main.py "$args"
done