#!/bin/bash
cd official/legacy/image_classification/resnet

python3 resnet_ctl_imagenet_main.py \
        --tpu=$TPU_NAME \
        --model_dir=$MODEL_DIR \
        --data_dir=$DATA_DIR \
        --batch_size=1024 \
        --steps_per_loop=500 \
        --train_epochs=$TRAIN_EPOCHS \
        --use_synthetic_data=false \
        --dtype=fp32 \
        --enable_eager=true \
        --enable_tensorboard=true \
        --distribution_strategy=tpu \
        --log_steps=50 \
        --single_l2_loss_op=true \
        --use_tf_function=true \
	--tiny
