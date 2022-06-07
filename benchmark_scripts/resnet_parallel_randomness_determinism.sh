#!/bin/bash
# Benchmark to ensure that enabling parallel randomness achieves
# determinism. This should be run with the custom-built tensorflow
# which comments out the tf.data optimizations.

# DATA_DIR is the inherited by top-level script,
# it's the root directory of the test data
IMAGENET_DIR=$DATA_DIR/imagenet2012/tfrecords

TRAIN_EPOCHS=10
NUM_ITERATIONS=3

NUM_IMAGES=1281167,50000
NUM_CLASSES=1001

DEFAULT_IMAGE_SIZE=224

RESIZE_MIN=256

for arg in "$@"
do
    if [ "$arg" == "--test" ]
    then
        echo "Using Tiny ImageNet, and reducing number of epochs."
        IMAGENET_DIR=$DATA_DIR/imagenet-tiny/tfrecords
        TRAIN_EPOCHS=20
        NUM_IMAGES=100000,10000
        NUM_CLASSES=200
        DEFAULT_IMAGE_SIZE=64
        RESIZE_MIN=$DEFAULT_IMAGE_SIZE
    fi
done


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
    "--deterministic"
)
echo "Using ${args[@]}"

cd official/legacy/image_classification/resnet

for i in $(seq $NUM_ITERATIONS)
do
    python3 resnet_ctl_imagenet_main.py "${args[@]}" --model_dir=$MODEL_DIR/parallel$i --parallel_randomness
done

for i in $(seq $NUM_ITERATIONS)
do
    python3 resnet_ctl_imagenet_main.py "${args[@]}" --model_dir=$MODEL_DIR/baseline$i
done