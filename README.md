# TensorFlow 教程示例

这个仓库是一个初级的TensorFlow使用示例，目的是方便初学者能够快速的熟悉TensorFlow，为了方便理解这些示例，这个仓库不经包含了代码，还有对应的jupyter notebook解释。

这个示例适合初学者，除了有原始TensorFlow的实现例子，还有最新的High Level API实现的例子，这些高级的API包含layers, Eager, estimator, dataset等

**更新 (03/18/2018):** TensorFlow开始支持Eager API (建议使用 TensorFlow v1.5+).

## 示例索引

#### 0 - 预习
- [机器学习简介](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/0_Prerequisite/ml_introduction.ipynb).
- [MNIST数据库简介](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/0_Prerequisite/mnist_dataset_intro.ipynb).

#### 1 - 简介
- **在TensorFlow中使用示例MNist数据集** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/ch/basic_operation/Baisc_operation_for_MNist.ipynb))
- **TensorFlow Hello World** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/ch/notebooks/1_Introduction/helloworld_ch.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/1_Introduction/helloworld.py)). 使用TensorFlow的最简单操作.
- **基本操作** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/ch/notebooks/1_Introduction/basic_operations_ch.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/1_Introduction/basic_operations.py)). 使用TensorFlow的基本操作.
- **TensorFlow Eager 基本操作** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/ch/notebooks/1_Introduction/basic_eager_api_ch.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/1_Introduction/basic_eager_api.py)).使用TensorFlow的 Eager API最基本操作.

#### 2 - 基本模型
- **线性回归** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/ch/notebooks/2_BasicModels/linear_regression_ch.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py)). 使用TensorFlow实现线性回归模型.
- **线性回归 (eager api)** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/ch/notebooks/2_BasicModels/linear_regression_eager_api_ch.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression_eager_api.py)). 使用TensorFlow的 Eager API实现线性回归.
- **逻辑回归** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/logistic_regression.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py)). 使用TensorFlow实现逻辑回归.
- **逻辑回归(eager api)** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/ch/notebooks/2_BasicModels/logistic_regression_ch.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression_eager_api.py)). 使用TensorFlow的 Eager API实现逻辑回归.
- **最近邻算法** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/nearest_neighbor.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/2_BasicModels/nearest_neighbor.py)). 使用 TensorFlow实现最近邻算法.
- **K-Means** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/ch/notebooks/2_BasicModels/kmeans_ch.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/2_BasicModels/kmeans.py)). 使用 TensorFlow 实现K-Means分类器.
- **随机森林** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/random_forest.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/2_BasicModels/random_forest.py)). 使用TensorFlow实现随机森林分类器.

#### 3 - 神经网络
##### 有监督

- **简单的神经网络** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/ch/notebooks/3_NeuralNetworks/neural_network_raw_ch.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network_raw.py)).使用TensorFlow原始的Low Level API实现简单的BP神经网络用于对MNist数据库进行分类.
- **简单的神经网络(tf.layers/estimator api)** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/ch/notebooks/3_NeuralNetworks/neural_network_ch.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network.py)).使用 TensorFlow 'layers' 和 'estimator' API 实现简单的神经网络实现对MNist数据库进行分类。
- **简单的神经网络(eager api)** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/neural_network_eager_api.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/neural_network_eager_api.py)). 使用 TensorFlow Eager API 构建一个多层感知机分类MNist数据库。
- **卷积神经网络（原始TensorFlow）** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/ch/notebooks/3_NeuralNetworks/convolutional_network_raw_ch.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py)). Build a convolutional neural network to classify MNIST digits dataset. Raw TensorFlow implementation.
- **Convolutional Neural Network (tf.layers/estimator api)** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py)). Use TensorFlow 'layers' and 'estimator' API to build a convolutional neural network to classify MNIST digits dataset.
- **Recurrent Neural Network (LSTM)** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/recurrent_network.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py)). Build a recurrent neural network (LSTM) to classify MNIST digits dataset.
- **Bi-directional Recurrent Neural Network (LSTM)** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/bidirectional_rnn.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py)). Build a bi-directional recurrent neural network (LSTM) to classify MNIST digits dataset.
- **Dynamic Recurrent Neural Network (LSTM)** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/dynamic_rnn.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py)). Build a recurrent neural network (LSTM) that performs dynamic calculation to classify sequences of different length.

##### 无监督
- **自动编码** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/autoencoder.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py)). Build an auto-encoder to encode an image to a lower dimension and re-construct it.
- **Variational Auto-Encoder** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/variational_autoencoder.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/variational_autoencoder.py)). Build a variational auto-encoder (VAE), to encode and generate images from noise.
- **GAN (Generative Adversarial Networks)** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/gan.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/gan.py)). Build a Generative Adversarial Network (GAN) to generate images from noise.
- **DCGAN (Deep Convolutional Generative Adversarial Networks)** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/dcgan.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dcgan.py)). Build a Deep Convolutional Generative Adversarial Network (DCGAN) to generate images from noise.

#### 4 - 实用工具
- **Save and Restore a model** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/4_Utils/save_restore_model.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/4_Utils/save_restore_model.py)). Save and Restore a model with TensorFlow.
- **Tensorboard - Graph and loss visualization** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/4_Utils/tensorboard_basic.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_basic.py)). Use Tensorboard to visualize the computation Graph and plot the loss.
- **Tensorboard - Advanced visualization** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/4_Utils/tensorboard_advanced.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_advanced.py)). Going deeper into Tensorboard; visualize the variables, gradients, and more...

#### 5 - 数据管理
- **Build an image dataset** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/5_DataManagement/build_an_image_dataset.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py)). Build your own images dataset with TensorFlow data queues, from image folders or a dataset file.
- **TensorFlow Dataset API** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/5_DataManagement/tensorflow_dataset_api.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/5_DataManagement/tensorflow_dataset_api.py)). Introducing TensorFlow Dataset API for optimizing the input data pipeline.

#### 6 - 多GPU
- **Basic Operations on multi-GPU** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/6_MultiGPU/multigpu_basics.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/6_MultiGPU/multigpu_basics.py)). A simple example to introduce multi-GPU in TensorFlow.
- **Train a Neural Network on multi-GPU** ([notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/6_MultiGPU/multigpu_cnn.ipynb)) ([code](https://github.com/debugCVML/TensorFlow-Examples/blob/master/examples/6_MultiGPU/multigpu_cnn.py)). A clear and simple TensorFlow implementation to train a convolutional neural network on multiple GPUs.

## 数据库
Some examples require MNIST dataset for training and testing. Don't worry, this dataset will automatically be downloaded when running examples.
MNIST is a database of handwritten digits, for a quick description of that dataset, you can check [this notebook](https://github.com/debugCVML/TensorFlow-Examples/blob/master/notebooks/0_Prerequisite/mnist_dataset_intro.ipynb).

Official Website: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

## 安装TensorFlow

To download all the examples, simply clone this repository:
```
git clone https://github.com/debugCVML/TensorFlow-Examples
```

To run them, you also need the latest version of TensorFlow. To install it:
```
pip install tensorflow
```

or (if you want GPU support):
```
pip install tensorflow_gpu
```

For more details about TensorFlow installation, you can check [TensorFlow Installation Guide](https://www.tensorflow.org/install/)

## 更多例程
The following examples are coming from [TFLearn](https://github.com/tflearn/tflearn), a library that provides a simplified interface for TensorFlow. You can have a look, there are many [examples](https://github.com/tflearn/tflearn/tree/master/examples) and [pre-built operations and layers](http://tflearn.org/doc_index/#api).

### 教程
- [TFLearn Quickstart](https://github.com/tflearn/tflearn/blob/master/tutorials/intro/quickstart.md). Learn the basics of TFLearn through a concrete machine learning task. Build and train a deep neural network classifier.

### 样例
- [TFLearn Examples](https://github.com/tflearn/tflearn/blob/master/examples). A large collection of examples using TFLearn.
