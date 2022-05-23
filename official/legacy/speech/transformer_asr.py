"""
Title: Automatic Speech Recognition with Transformer
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2021/01/13
Last modified: 2021/01/13
Description: Training a sequence-to-sequence Transformer for automatic speech recognition.
"""
"""
## Introduction

Automatic speech recognition (ASR) consists of transcribing audio speech segments into text.
ASR can be treated as a sequence-to-sequence problem, where the
audio can be represented as a sequence of feature vectors
and the text as a sequence of characters, words, or subword tokens.

For this demonstration, we will use the LJSpeech dataset from the
[LibriVox](https://librivox.org/) project. It consists of short
audio clips of a single speaker reading passages from 7 non-fiction books.
Our model will be similar to the original Transformer (both encoder and decoder)
as proposed in the paper, "Attention is All You Need".


**References:**

- [Attention is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Very Deep Self-Attention Networks for End-to-End Speech Recognition](https://arxiv.org/pdf/1904.13377.pdf)
- [Speech Transformers](https://ieeexplore.ieee.org/document/8462506)
- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
"""
import json
from pathlib import Path

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from official.legacy.image_classification.classifier_trainer import (
    define_classifier_flags,
)
from official.legacy.speech import callbacks, dataset, layers
from official.utils.misc.keras_utils import TimeHistory


def create_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices("TPU"))

    return tf.distribute.TPUStrategy(resolver)


def run(flags_obj):
    strategy = create_strategy()
    max_target_len = 200  # all transcripts in out data are < 200 characters

    vectorizer = dataset.VectorizeChar(max_len=max_target_len)
    ds_factory = dataset.DatasetFactory(vectorizer)
    # ds, val_ds = [strategy.distribute_datasets_from_function(lambda _: ds_factory.get_dataset(is_training)) for is_training in [True, False]]
    ds, val_ds = (ds_factory.get_dataset(is_training) for is_training in [True, False])

    with strategy.scope():
        model = layers.create_model(max_target_len)

    time_history = TimeHistory(64, flags_obj.log_steps, logdir=flags_obj.model_dir)
    model.fit(
        ds,
        validation_data=val_ds,
        callbacks=[
            # callbacks.DisplayOutputs(
            #     next(iter(val_ds)),
            #     vectorizer.get_vocabulary(),
            #     target_start_token_idx=2,
            #     target_end_token_idx=3,
            # ),
            time_history,
        ],
        epochs=10,
        # steps_per_epoch=202,  # from observation
    )

    return {
        "epoch_runtime_log": time_history.epoch_runtime_log,
        "batch_runtime_log": time_history.batch_runtime_log,
    }


"""
In practice, you should train for around 100 epochs or more.

Some of the predicted text at or around epoch 35 may look as follows:
```
target:     <as they sat in the car, frazier asked oswald where his lunch was>
prediction: <as they sat in the car frazier his lunch ware mis lunch was>

target:     <under the entry for may one, nineteen sixty,>
prediction: <under the introus for may monee, nin the sixty,>
```
"""


def main(_):
    if flags.FLAGS.enable_op_determinism:
        tf.keras.utils.set_random_seed(1)
        tf.config.experimental.enable_op_determinism()
    stats = run(flags.FLAGS)
    if stats:
        logging.info("Run stats:\n%s", stats)
        with Path(flags.FLAGS.model_dir, "stats.json").open("w") as f:
            json.dump(stats, f, default=str, indent=2)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    define_classifier_flags()
    # flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("model_dir")
    app.run(main)
