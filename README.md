<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>

-----------------

# Tensorflow with KenLM integration

This fork of tensorflow adds KenLM on tf version 1.2.1 and is based from https://github.com/timediv/tensorflow-with-kenlm

Only modification done on top of this "https://github.com/timediv/tensorflow-with-kenlm" is that it is integrated with the TF 1.2.1

-----------------

This fork of tensorflow adds [KenLM](http://kheafield.com/professional/edinburgh/estimate_paper.pdf) (a language model) to the `ctc_beam_search_decoder` operation.

```
tf.nn.ctc_beam_search_decoder(logits,
                              output_sequence_lengths,
                              kenlm_directory_path='your/directory/path')
```

Your specified kenlm_directory_path must contain three files
```
kenlm-model.binary
vocabulary
trie
```

See http://kheafield.com/code/kenlm/ to find out how to generate your `kenlm-model.binary`.

The `vocabulary` file contains the mapping from your logit labels to characters,
the file should contain all allowed characteres in a single line,
the indexing specifying the respective label id, e.g.
```
abcdefghijklmnopqrstuvwxyz '
```

The `trie` is generated from a text corpus of all words on a character level.
Given a file `corpus.txt` which must satisfy the following conditions,
- only contains words with characters specified in `vocabulary`
- seperated by whitespace or new lines

we can generate `trie` using:
```
cd tensorflow-with-kenlm
bazel build -c opt --config=cuda //tensorflow/core/util/ctc:ctc_generate_trie
bazel-bin/tensorflow/core/util/ctc/ctc_generate_trie kenlm-model.binary vocabulary < corpus.txt > trie
```

## How to compile tensorflow

See [Download and Setup](tensorflow/g3doc/get_started/os_setup.md) for more detailed instructions.
```
./configure
bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-*.whl --upgrade
```

-----------------

| **`Linux CPU`** | **`Linux GPU`** | **`Mac OS CPU`** | **`Windows CPU`** | **`Android`** |
|-----------------|---------------------|------------------|-------------------|---------------|
| [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-cpu)](https://ci.tensorflow.org/job/tensorflow-master-cpu) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-linux-gpu)](https://ci.tensorflow.org/job/tensorflow-master-linux-gpu) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-mac)](https://ci.tensorflow.org/job/tensorflow-master-mac) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-win-cmake-py)](https://ci.tensorflow.org/job/tensorflow-master-win-cmake-py) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-android)](https://ci.tensorflow.org/job/tensorflow-master-android) |

**TensorFlow** is an open source software library for numerical computation using
data flow graphs.  Nodes in the graph represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them.  This flexible architecture lets you deploy computation to one
or more CPUs or GPUs in a desktop, server, or mobile device without rewriting
code.  TensorFlow also includes TensorBoard, a data visualization toolkit.

TensorFlow was originally developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence research
organization for the purposes of conducting machine learning and deep neural
networks research.  The system is general enough to be applicable in a wide
variety of other domains, as well.

**If you'd like to contribute to TensorFlow, be sure to review the [contribution
guidelines](CONTRIBUTING.md).**

**We use [GitHub issues](https://github.com/tensorflow/tensorflow/issues) for
tracking requests and bugs, but please see
[Community](https://www.tensorflow.org/community/) for general questions
and discussion.**

## Installation
*See [Installing TensorFlow](https://www.tensorflow.org/install/) for instructions on how to install our release binaries or how to build from source.*

People who are a little more adventurous can also try our nightly binaries:

* Linux CPU-only: [Python 2](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-1.2.1-cp27-none-linux_x86_64.whl) ([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave)) / [Python 3.4](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-1.2.1-cp34-cp34m-linux_x86_64.whl) ([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=cpu-slave/)) / [Python 3.5](https://ci.tensorflow.org/view/Nightly/job/nightly-python35-linux-cpu/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-1.2.1-cp35-cp35m-linux_x86_64.whl) ([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-python35-linux-cpu/))
* Linux GPU: [Python 2](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-linux-gpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=gpu-linux/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow_gpu-1.2.1-cp27-none-linux_x86_64.whl) ([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-linux-gpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=gpu-linux/)) / [Python 3.4](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-linux-gpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=gpu-linux/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow_gpu-1.2.1-cp34-cp34m-linux_x86_64.whl) ([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-linux-gpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=gpu-linux/)) / [Python 3.5](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-linux-gpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=gpu-linux/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow_gpu-1.2.1-cp35-cp35m-linux_x86_64.whl) ([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-linux-gpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=gpu-linux/))
* Mac CPU-only: [Python 2](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=mac-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-1.2.1-py2-none-any.whl) ([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=mac-slave/)) / [Python 3](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=mac-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-1.2.1-py3-none-any.whl) ([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=mac-slave/))
* Mac GPU: [Python 2](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-mac-gpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=gpu-mac/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow_gpu-1.2.1-py2-none-any.whl) ([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-mac-gpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=gpu-mac/)) / [Python 3](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-mac-gpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=gpu-mac/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow_gpu-1.2.1-py3-none-any.whl) ([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-mac-gpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=gpu-mac/))
* Windows CPU-only: [Python 3.5 64-bit](https://ci.tensorflow.org/view/Nightly/job/nightly-win/M=windows,PY=35/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tensorflow-1.2.1-cp35-cp35m-win_amd64.whl) ([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-win/M=windows,PY=35/)) / [Python 3.6 64-bit](https://ci.tensorflow.org/view/Nightly/job/nightly-win/M=windows,PY=36/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tensorflow-1.2.1-cp36-cp36m-win_amd64.whl) ([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-win/M=windows,PY=36/))
* Windows GPU: [Python 3.5 64-bit](https://ci.tensorflow.org/view/Nightly/job/nightly-win/M=windows-gpu,PY=35/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tensorflow_gpu-1.2.1-cp35-cp35m-win_amd64.whl) ([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-win/M=windows-gpu,PY=35/)) / [Python 3.6 64-bit](https://ci.tensorflow.org/view/Nightly/job/nightly-win/M=windows-gpu,PY=36/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tensorflow_gpu-1.2.1-cp36-cp36m-win_amd64.whl) ([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-win/M=windows-gpu,PY=36/))
* Android: [demo APK](https://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/tensorflow_demo.apk), [native libs](http://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/native/)
([build history](https://ci.tensorflow.org/view/Nightly/job/nightly-android/))

#### *Try your first TensorFlow program*
```shell
$ python
```
```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
'Hello, TensorFlow!'
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a+b)
42
>>>
```

## For more information

* [TensorFlow website](https://tensorflow.org)
* [TensorFlow whitepaper](http://download.tensorflow.org/paper/whitepaper2015.pdf)
* [TensorFlow Model Zoo](https://github.com/tensorflow/models)
* [TensorFlow MOOC on Udacity](https://www.udacity.com/course/deep-learning--ud730)

The TensorFlow community has created amazing things with TensorFlow, please see the [resources section of tensorflow.org](https://www.tensorflow.org/about/#community) for an incomplete list.
