#!/bin/bash
cd official/legacy/speech
python transformer_asr.py --dataset_name="ljspeech" --data_dir="LJSpeech-1.1" --max_target_len=2048 --model_dir="${MODEL_DIR?}" "$@"