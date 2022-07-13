#!/bin/bash

# DATA_DIR is the inherited by top-level script,
# it's the root directory of the test data
cd official/benchmark/models

for i in {1..5}
do
	python resnet_cifar_main.py \
		--data_dir /home/zkck/training-data/imagenet2012/tfrecords \
		--tpu "local" \
		-ds tpu \
		--train_epochs 90 \
		--enable_tensorboard \
		--model_dir "/home/zkck/data/run$i" \
		--seed 1 \
		--enable_op_determinism
done
