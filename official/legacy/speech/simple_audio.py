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
import tensorflow_datasets as tfds
from official.utils.misc.keras_utils import TimeHistory
from tensorflow.keras import layers
from tensorflow.keras import models
from absl import flags
from absl import app
from absl import logging



NUM_LABELS = 8

EPOCHS = 10


def create_tpu_strategy():
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
  tf.config.experimental_connect_to_cluster(resolver)
  # This is the TPU initialization code that has to be at the beginning.
  tf.tpu.experimental.initialize_tpu_system(resolver)
  print("All devices: ", tf.config.list_logical_devices('TPU'))

  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

  with tf.device('/TPU:0'):
    c = tf.matmul(a, b)

  print("c device: ", c.device)
  print(c)

  return tf.distribute.TPUStrategy(resolver)


def create_model(dataset):
  # # Instantiate the `tf.keras.layers.Normalization` layer.
  # norm_layer = layers.Normalization()
  # # Fit the state of the layer to the spectrograms
  # # with `Normalization.adapt`.
  # norm_layer.adapt(data=dataset.map(map_func=lambda spec, label: spec))
  return models.Sequential([
      layers.Input(shape=(124, 129, 1)),
      # Downsample the input.
      layers.Resizing(32, 32),
      # Normalize.
      # norm_layer,
      layers.Conv2D(32, 3, activation='relu'),
      layers.Conv2D(64, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.25),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(NUM_LABELS),
  ])

def build_stats(time_history):
  return {
    'epoch_runtime_log': time_history.epoch_runtime_log,
    'batch_runtime_log': time_history.batch_runtime_log,
  }



def run(flags_obj):
  strategy = create_tpu_strategy()

  train_ds = tfds.load('speech_commands', split='train')
  val_ds = tfds.load('speech_commands', split='validation')

  with strategy.scope():
    model = create_model(train_ds)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

  batch_size = 64
  train_ds = train_ds.batch(batch_size)
  val_ds = val_ds.batch(batch_size)

  train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
  val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

  time_history = TimeHistory(batch_size=batch_size, log_steps=flags_obj.log_steps)

  model.fit(
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

