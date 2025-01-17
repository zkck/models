This repository contains modified ML training pipeline benchmarks for evaluating the implementation of [per-element seeds in TensorFlow](http://hdl.handle.net/20.500.11850/525189), a solution for providing parallel reproducible randomness (PRR) while retaining stateful operations. Our benchmarks consist of two parts: one being the motivation behind PRR, i.e. the slowdown that `enable_op_determinism` may incur, the other being the evaluation of PRR by validating the functionality and measuring performance overhead.

For technical setup, we used TPU VMs and datasets on attached SSD mounted at `~/training-data/`.

The modifications of the benchmarks include among others:

- Changing weight initialization to deterministic behaviors
- Scripts for running benchmarks under `benchmark_scripts/`
- Modifications to include heavier input pipelines
- Conditional statements in the input pipeline for use PRR
- Changes for compatibility with datasets
- Saving additional run statistics

The models/datasets that we use in our motivation for analyzing performance overhead of `enable_op_determinism` are as follows, each found in `benchmark_scripts/` (see branches below):

| Model | Dataset |
| ------ | ----------- |
| ResNet50 | ImageNet |
| ResNet50 | CIFAR10 |
| EfficientNet | ImageNet |
| EfficientNet | CIFAR10 |
| RetinaNet | COCO |
| Transformer | LJSpeech (+ SpecAugment) |
| Transformer | LibriSpeech (+ SpecAugment) |

The models/datasets that we use in our evaluation are as follows:

| Model | Dataset |
| ------ | ----------- |
| ResNet50 | ImageNet |
| ResNet52 | CIFAR10 |
| EfficientNet | ImageNet |
| RetinaNet | COCO |

There are many branches in this repository, most of which were used for debugging. The following table lists the main branches used for benchmarking, and a description describing what purpose they served in the evaluation of PRR.

| Branch | Description |
| ------ | ----------- |
| zacook/motivation<br />zacook/accuracy | Branches used for evaluating the overhead of `enable_op_determinism`, with scripts located in `benchmark_scripts/` |
| zacook/deterministic-weights | Branch used for testing PRR, requiring removing all sources of non-determinism except for those due to parallelism with stateful operations |

To run a benchmark in `benchmark_scripts`, make sure that datasets are mounted in `~/training-data/` and that the Model Garden codebase is installed with `PYTHONPATH=/path/to/models`, and run the benchmark script from the root of this codebase. The layout of the training data directory should be as follows:

```
zacook@zacook-instance-1:~$ tree training-data/ -d --filelimit 100
training-data/
├── cache_temp
├── cifar10
│   └── 3.0.2
├── coco
│   └── raw-data
│       ├── annotations
│       ├── test2017
│       ├── train2017
│       └── val2017
├── imagenet-tiny
│   ├── tfrecords
│   │   ├── train
│   │   └── validation
│   └── tiny-imagenet-200
│       ├── test
│       │   └── images
│       ├── train [200 entries exceeds filelimit, not opening dir]
│       └── validation
│           └── images
├── imagenet2012
│   ├── ILSVRC2012_devkit_t12
│   │   ├── data
│   │   └── evaluation
│   ├── tfrecords
│   │   ├── train
│   │   └── validation
│   ├── train [1000 entries exceeds filelimit, not opening dir]
│   └── val [1000 entries exceeds filelimit, not opening dir]
...
```

From this line onwards is the original README.

![Logo](https://storage.googleapis.com/model_garden_artifacts/TF_Model_Garden.png)

# Welcome to the Model Garden for TensorFlow

The TensorFlow Model Garden is a repository with a number of different implementations of state-of-the-art (SOTA) models and modeling solutions for TensorFlow users. We aim to demonstrate the best practices for modeling so that TensorFlow users
can take full advantage of TensorFlow for their research and product development.

| Directory | Description |
|-----------|-------------|
| [official](official) | • A collection of example implementations for SOTA models using the latest TensorFlow 2's high-level APIs<br />• Officially maintained, supported, and kept up to date with the latest TensorFlow 2 APIs by TensorFlow<br />• Reasonably optimized for fast performance while still being easy to read |
| [research](research) | • A collection of research model implementations in TensorFlow 1 or 2 by researchers<br />• Maintained and supported by researchers |
| [community](community) | • A curated list of the GitHub repositories with machine learning models and implementations powered by TensorFlow 2 |

## [Announcements](https://github.com/tensorflow/models/wiki/Announcements)

| Date | News |
|------|------|
| June 17, 2020 | [Context R-CNN: Long Term Temporal Context for Per-Camera Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection#june-17th-2020) released
| May 21, 2020 | [Unifying Deep Local and Global Features for Image Search (DELG)](https://github.com/tensorflow/models/tree/master/research/delf#delg) code released
| May 19, 2020 | [MobileDets: Searching for Object Detection Architectures for Mobile Accelerators](https://github.com/tensorflow/models/tree/master/research/object_detection#may-19th-2020) released
| May 7, 2020 | [MnasFPN with MobileNet-V2 backbone](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#mobile-models) released for object detection
| May 1, 2020 | [DELF: DEep Local Features](https://github.com/tensorflow/models/tree/master/research/delf) updated to support TensorFlow 2.1
| March 31, 2020 | [Introducing the Model Garden for TensorFlow 2](https://blog.tensorflow.org/2020/03/introducing-model-garden-for-tensorflow-2.html) ([Tweet](https://twitter.com/TensorFlow/status/1245029834633297921)) |

## [Milestones](https://github.com/tensorflow/models/milestones)

| Date | Milestone |
|------|-----------|
| July 7, 2020 | [![GitHub milestone](https://img.shields.io/github/milestones/progress/tensorflow/models/1)](https://github.com/tensorflow/models/milestone/1) |

## Contributions

[![help wanted:paper implementation](https://img.shields.io/github/issues/tensorflow/models/help%20wanted%3Apaper%20implementation)](https://github.com/tensorflow/models/labels/help%20wanted%3Apaper%20implementation)

If you want to contribute, please review the [contribution guidelines](https://github.com/tensorflow/models/wiki/How-to-contribute).

## License

[Apache License 2.0](LICENSE)
