#!/bin/bash

suffix="${1?}"

for i in {1..5}
do
	TRAIN_EPOCHS="${TRAIN_EPOCHS?}" MODEL_DIR="$HOME/data-imagenet-$suffix/run$i" bash run.sh
done
