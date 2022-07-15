#!/bin/bash

suffix="${1?}"
train_epochs="${TRAIN_EPOCHS?}"

for i in {1..5}
do
	model_dir="$HOME/data-imagenet-$suffix/run$i"
	rm -rf "$model_dir"
	TRAIN_EPOCHS="$train_epochs" MODEL_DIR="$model_dir" bash run.sh
done
