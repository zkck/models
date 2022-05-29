import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp


def mask_frequency(x, frequency_masking_para):
    with tf.name_scope('mask_frequency'):
        max_freq = tf.shape(x)[1]
        f = tf.random.uniform(shape=(), minval=0, maxval=frequency_masking_para, dtype=tf.dtypes.int32)
        f0 = tf.random.uniform(
            shape=(), minval=0, maxval=max_freq - f, dtype=tf.dtypes.int32
        )
        indices = tf.reshape(tf.range(max_freq), (1, -1))
        condition = tf.math.logical_and(
            tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
        )
        return tf.where(condition, 0., x)


def mask_time(x, time_masking_para):
    with tf.name_scope('mask_time'):
        tau = tf.shape(x)[0]
        t = tf.random.uniform(shape=(), minval=0, maxval=time_masking_para, dtype=tf.dtypes.int32)
        t0 = tf.random.uniform(
            shape=(), minval=0, maxval=tau - t, dtype=tf.dtypes.int32
        )
        indices = tf.reshape(tf.range(tau), (-1, 1))
        condition = tf.math.logical_and(
            tf.math.greater_equal(indices, t0), tf.math.less(indices, t0 + t)
        )
        return tf.where(condition, 0., x)


def time_warp(x, time_warping_para):
    tau = tf.shape(x)[0]
    height = tf.shape(x)[1]

    # Step 1 : Time warping
    if tau <= 2 * time_warping_para:
        tf.print("Spectrogram too short (len = ", tau, ") for time_warping_param = ", time_warping_para,
                 ". Skipping warping.")
        return x

    # tf.assert_greater(tau, 2 * time_warping_para)
    # if tau < 2 * time_warping_para:
    # tf.print((tf.shape(x), time_warping_para, tau - time_warping_para))
    # tf.print(time_warping_para)
    generator = tf.random.get_global_generator()
    center_height = height / 2

    with tf.name_scope('random_point'):
        random_point = generator.uniform(minval=time_warping_para, maxval=tau - time_warping_para, dtype=tf.int32,
                                         shape=(),
                                         name='get_random_point')
    with tf.name_scope('warping'):
        w = generator.uniform(minval=0, maxval=time_warping_para, dtype=tf.int32, shape=(), name='get_warping_factor')

    control_point_locations = tf.convert_to_tensor([[[random_point, center_height],
                                                     [0, center_height],
                                                     [tau, center_height]]],
                                                   dtype=tf.float32)

    control_point_destination = tf.convert_to_tensor([[[random_point + w, center_height],
                                                       [0, center_height],
                                                       [tau, center_height]]],
                                                     dtype=tf.float32)
    ret, _ = sparse_image_warp(x,
                               source_control_point_locations=control_point_locations,
                               dest_control_point_locations=control_point_destination,
                               interpolation_order=2,
                               regularization_weight=0,
                               num_boundary_points=1
                               )

    return ret


def spec_augment(x, time_warping_para=80, frequency_masking_para=27,
                 time_masking_para=100):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      x (numpy array): spectrogram.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      # not implemented yet:
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.

    # Returns
      x (numpy array): warped and masked mel spectrogram.
    """
    with tf.name_scope('specaugment'):
        tau = tf.shape(x)[0]

        # Step 1 : Time warping
        if tau > 3 * time_warping_para:  # Warp only if audio is long enough.
            x = time_warp(x, time_warping_para)

        # Step 2 : Frequency masking
        x = mask_frequency(x, frequency_masking_para)

        # Step 3 : Time masking
        if tau > 3 * time_masking_para:  # Warp only if audio is long enough.
            x = mask_time(x, time_masking_para)
    return x
