import tensorflow as tf

def split_uppercase(string):
    i = 0
    for j in range(1, len(string)):
        if string[j].isupper():
            yield string[i:j]
            i = j
    yield string[i:]

class RandomNormal(tf.keras.initializers.RandomNormal):

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
      super().__init__(mean=mean, stddev=stddev, seed=seed)
      self._random_generator._force_generator = True


class HeNormal(tf.keras.initializers.VarianceScaling):

  def __init__(self, seed=None):
    super(HeNormal, self).__init__(
        scale=2., mode='fan_in', distribution='truncated_normal', seed=seed)
    self._random_generator._force_generator = True

  def get_config(self):
    return {'seed': self.seed}


class VarianceScaling(tf.keras.initializers.VarianceScaling):
  def __init__(self,
               scale=1.0,
               mode='fan_in',
               distribution='truncated_normal',
               seed=None):
      super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed)
      self._random_generator._force_generator = True


class DeterministicInitializerFactory:

  _INITIALIZERS = {
    'he_normal': HeNormal,
    'normal': RandomNormal,
    'variance_scaling': VarianceScaling,
  }

  def __init__(self, seed) -> None:
      self.g = tf.random.Generator.from_seed(seed)

  def make_initializer(self, initializer_type, **kwargs):
      # Convert 'VarianceScaling' to 'variance_scaling'
      initializer_type = '_'.join(split_uppercase(initializer_type))
      if initializer_type not in self._INITIALIZERS:
        raise ValueError(f"Initializer type {initializer_type} not found.")
      return self._INITIALIZERS[initializer_type](seed=self.g.uniform_full_int([]), **kwargs)

_INITIALIZER_FACTORY = DeterministicInitializerFactory(66)
