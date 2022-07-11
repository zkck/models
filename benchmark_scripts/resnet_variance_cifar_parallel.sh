#!/bin/bash
export ZCK_PARALLEL_RANDOMNESS=1
cd official/benchmark/models

for i in {1..5}
do
	python resnet_cifar_main.py --data_dir /home/zkck/cifar-10-batches-bin/ --tpu local -ds tpu --train_epochs 200 --enable_tensorboard --model_dir /home/zkck/data/run$i --seed 1
done
