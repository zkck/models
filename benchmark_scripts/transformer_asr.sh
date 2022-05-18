#!/bin/bash
cd official/legacy/speech
python transformer_asr.py --model_dir="${MODEL_DIR?}" "$@"