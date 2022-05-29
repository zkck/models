#!/bin/bash
cd official/legacy/speech
python transformer_asr.py --dataset_name=librispeech --data_dir="LibriSpeech/train-clean-100" --max_target_len=2048 --model_dir="${MODEL_DIR?}" "$@"