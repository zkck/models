#!/bin/bash

suffix="${1?}"
model_dir="$HOME/data-effnet-$suffix"
rm -rf "$model_dir"

for i in {1..5}
do
	run_dir="$model_dir/run$i"
	MODEL_DIR="$run_dir" bash run.sh
done
