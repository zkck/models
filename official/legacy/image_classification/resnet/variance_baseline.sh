#!/bin/bash
export ZCK_PARALLEL_RANDOMNESS=1
TRAIN_EPOCHS="${TRAIN_EPOCHS?}" bash variance.sh per-element-seeds
