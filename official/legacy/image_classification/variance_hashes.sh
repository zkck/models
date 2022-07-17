#!/bin/bash
export ZCK_CHECK_HASHES=1
export ZCK_PARALLEL_RANDOMNESS=1
export PYTHONHASHSEED=1
bash variance.sh hashes
