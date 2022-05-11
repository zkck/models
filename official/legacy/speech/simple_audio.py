# Copyright 2020 The TensorFlow Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import pathlib

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from official.common import distribute_utils

from official.legacy.image_classification.classifier_trainer import define_classifier_flags

from official.legacy.speech import preprocessing
import tensorflow as tf
from official.utils.misc.keras_utils import TimeHistory
from tensorflow.keras import layers
from tensorflow.keras import models
from absl import flags
from absl import app
from absl import logging



NUM_LABELS = 8

EPOCHS = 10


def make_model(dataset):
  for spectrogram, _ in dataset.take(1):
    input_shape = spectrogram.shape
  # Instantiate the `tf.keras.layers.Normalization` layer.
  norm_layer = layers.Normalization()
  # Fit the state of the layer to the spectrograms
  # with `Normalization.adapt`.
  norm_layer.adapt(data=dataset.map(map_func=lambda spec, label: spec))

  model = models.Sequential([
      layers.Input(shape=input_shape),
      # Downsample the input.
      layers.Resizing(32, 32),
      # Normalize.
      norm_layer,
      layers.Conv2D(32, 3, activation='relu'),
      layers.Conv2D(64, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.25),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(NUM_LABELS),
  ])

  model.summary()


  model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
  )

  return model


def build_stats(time_history):
  return {
    'epoch_runtime_log': time_history.epoch_runtime_log,
    'batch_runtime_log': time_history.batch_runtime_log,
  }

def run(flags_obj):
  distribute_utils.configure_cluster(flags_obj.worker_hosts,
                                     flags_obj.task_index)

  # Note: for TPUs, strategy and scope should be created before the dataset
  strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy='tpu',
      tpu_address='local')

  strategy_scope = distribute_utils.get_strategy_scope(strategy)

  logging.info('Detected %d devices.',
               strategy.num_replicas_in_sync if strategy else 1)


  with strategy_scope:
    train_ds, val_ds, test_ds = preprocessing.make_datasets(pathlib.Path(flags_obj.data_dir))
    model = make_model(train_ds)

  batch_size = 64
  train_ds = train_ds.batch(batch_size)
  val_ds = val_ds.batch(batch_size)

  train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
  val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

  time_history = TimeHistory(batch_size=batch_size, log_steps=flags_obj.log_steps)

  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
      time_history,
      tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)],
  )

  return build_stats(time_history)


def main(_):
  if flags.FLAGS.enable_op_determinism:
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
  stats = run(flags.FLAGS)
  if stats:
    model_dir = pathlib.Path(flags.FLAGS.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    logging.info('Run stats:\n%s', stats)
    with (model_dir / 'stats.json').open('w') as f:
      json.dump(stats, f, default=str, indent=2)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_classifier_flags()
  flags.mark_flag_as_required('data_dir')
  flags.mark_flag_as_required('model_dir')
  app.run(main)

