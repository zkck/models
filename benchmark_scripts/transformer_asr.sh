#!/bin/bash
cd official/legacy/speech
python transformer_asr.py --max_target_len=2048 --model_dir="${MODEL_DIR?}" "$@"