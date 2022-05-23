import glob
import os
import tensorflow as tf
from tensorflow import keras


def path_to_audio(path):
    # spectrogram using stft
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)
    audio = tf.squeeze(audio, axis=-1)
    stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / stddevs
    audio_len = tf.shape(x)[0]
    # padding to 10 seconds
    pad_len = 2754
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x


class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
            ["-", "#", "<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", ".", ",", "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab


class DatasetFactory:
    def __init__(self, vectorizer: VectorizeChar) -> None:
        self._vectorizer = vectorizer
        self._data = []
        self._id_to_text = {}
        self.__populate()

    def __populate(self):
        keras.utils.get_file(
            os.path.join(os.getcwd(), "data.tar.gz"),
            "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
            extract=True,
            archive_format="tar",
            cache_dir=".",
        )

        saveto = "./datasets/LJSpeech-1.1"
        wavs = glob.glob("{}/**/*.wav".format(saveto), recursive=True)

        with open(os.path.join(saveto, "metadata.csv"), encoding="utf-8") as f:
            for line in f:
                id = line.strip().split("|")[0]
                text = line.strip().split("|")[2]
                self._id_to_text[id] = text

        for w in wavs:
            id = w.split("/")[-1].split(".")[0]
            if len(self._id_to_text[id]) < self._vectorizer.max_len:
                self._data.append({"audio": w, "text": self._id_to_text[id]})

    def create_text_ds(self, data):
        texts = [_["text"] for _ in data]
        text_ds = [self._vectorizer(t) for t in texts]
        text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
        return text_ds

    def create_audio_ds(self, data):
        flist = [_["audio"] for _ in data]
        audio_ds = tf.data.Dataset.from_tensor_slices(flist)
        audio_ds = audio_ds.map(path_to_audio, num_parallel_calls=tf.data.AUTOTUNE)
        return audio_ds

    def create_tf_dataset(self, data, bs=4):
        audio_ds = self.create_audio_ds(data)
        text_ds = self.create_text_ds(data)
        ds = tf.data.Dataset.zip((audio_ds, text_ds))
        ds = ds.map(lambda x, y: {"source": x, "target": y}, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(bs, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def get_dataset(self, is_training):
        print("vocab size", len(self._vectorizer.get_vocabulary()))
        print("data size", len(self._data))
        split = int(len(self._data) * 0.99)
        if is_training:
            data, bs = self._data[:split], 64
        else:
            data, bs = self._data[split:], 4
        return self.create_tf_dataset(data, bs=bs)
