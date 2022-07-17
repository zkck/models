#!/bin/bash
export ZCK_CHECK_WEIGHTS=1
export ZCK_PARALLEL_RANDOMNESS=1
export PYTHONHASHSEED=1
bash variance.sh weights
