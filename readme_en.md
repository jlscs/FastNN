# FastNN Model Library
## 1. Introduction
FastNN(Fast Neural Networks)aims to show how to implement distributed model based on [PAISoar](https://yq.aliyun.com/articles/705132), which allows researchers to most effective apply distributed neural networks. For now, FastNN only provides some classic models on Computer vision, but is preparing state-of-art models on natural language processing.

If you gonna to try out on PAI(Platform of Artificial Intelligence)for distributed FastNN, please turn to [PAI Homepage](https://data.aliyun.com/product/learn?spm=5176.12825654.eofdhaal5.143.2cc52c4af9oxZf), then submit machine-learning jobs on PAI Studio or DSW-notebook. Relative instructions are clarified in [TensorFlow manual](https://help.aliyun.com/document_detail/49571.html?spm=a2c4g.11186623.6.579.10501312JxztvO), and the following second "Quick Start" chapter.

FastNN Features：
* Models

    a.Some classic models on computer vision，including inception、resnet、mobilenet、vgg、alexnet、nasnet and so on;
    
    b.Preparing state-of-art models on natural language processing, including Bert、XLnet、NMT、GPT-2;
    
* Distributed Plans（Turn to PAI for submitting jobs on PAI Studio or DSW-notebook）

    a.Single-Node Multi-GPUs
    
    b.Multi-Nodes Multi-GPUs
    
* Half-Precision Training
* Task Type

    a.Model Pretrain
    
    b.Model Finetune：Restore only trainable variables default. Gonna to self-defined checkpoint restoring，please turn to get_assigment_map_from_checkpoint in file "images/utils/misc_utils.py".


We choose ResNet-v1-50 model and conduct large-scale test on Alibaba Cloud Computing Cluster(GPU P100). As the chart shows，PAISoar performs perfectly with nearly linear acceleration.

![resnet_v1_50](https://pai-online.oss-cn-shanghai.aliyuncs.com/fastnn-data/readme/resnet_v1_50.png)

## 2. Quick Start
This chaper is all about intructions on FastNN usage without any code modification, including：
* Data Preparing: Preparing local or PAI-Web training data;
* Kick-off training: including setting for local shell script or PAI Web training parameters。

### 2.1 Data Preparing
For the convenience of trying out computer vision models in FastNN model Library, we prepare some open datasets (including cifar10、mnist and flowers) or their relative download_and_convert shell scripts.
#### 2.1.1 Local datasets

Learning from TF-Slim model library, we provide script（images/datasets/download_and_convert_data.py）for downloading and converting to TFRecord format. Take cifar10 for example, script as following:
```
DATA_DIR=/tmp/data/cifar10
python download_and_convert_data.py \
	--dataset_name=cifar10 \
	--dataset_dir="${DATA_DIR}"
```
After script executed，we will get the following tfrecord files in /tmp/data/cifar10:
>$ ls ${DATA_DIR}

>cifar10_train.tfrecord

>cifar10_test.tfrecord

>labels.txt

#### 2.1.2 OSS dataset

For the convenience of trying out FastNN model library on [PAI](https://data.aliyun.com/product/learn?spm=5176.12825654.eofdhaal5.143.2cc52c4af9oxZf), we already download and convert some datasets into TFRecord format, including cifar10、mnist、flowers, and can be accessed by oss api in PAI, oss path shows as:

|dataset|num of classes|training set|test set|storage path|
| :-----: | :----: | :-----:| :----:| :---- |
| mnist   |  10    |  3320  | 350   | BeiJing：oss://pai-online-beijing.oss-cn-beijing-internal.aliyuncs.com/fastnn-data/mnist/ ShangHai：oss://pai-online.oss-cn-shanghai-internal.aliyuncs.com/fastnn-data/mnist/
| cifar10 |  10    | 50000  |10000  | BeiJing：oss://pai-online-beijing.oss-cn-beijing-internal.aliyuncs.com/fastnn-data/cifar10/ ShangHai：oss://pai-online.oss-cn-shanghai-internal.aliyuncs.com/fastnn-data/cifar10/
| flowers |  5     |60000   |10000  | BeiJing：oss://pai-online-beijing.oss-cn-beijing-internal.aliyuncs.com/fastnn-data/flowers/ ShangHai：oss://pai-online.oss-cn-shanghai-internal.aliyuncs.com/fastnn-data/flowers/

### 2.2 Kick-off Training
The main file in FastNN is 'train_image_classifiers.py'. User parameters, Model base parameters as well as relative instructions are summaried in file 'flags.py'. 
For more information, turn to chapter 3. Among all params, the most common six params are listed as the following:
* task_type：String type. options among ['pretrain', 'finetune'], which clarifies task is 'pretrain' or 'finetune', default 'pretrain';
* enable_paisoar：Bool type. Default True，when trying out locally, should set False;
* dataset_name：String type. Indicating training dataset, like files 'cifar10.py、flowers.py、mnist.py' in 'images/datasets', default mock;
* train_files：String type. Indicating names of all training files separated by comma, default None;
* dataset_dir：String type. Indicating training dataset directory, default None;
* model_name：String type. Indicating model name, options among ['resnet_v1_50', 'vgg', 'inception']，for more info, check images/models;
Particularly, if task_type is 'finetune', model_dir and ckpt_file_name need to be assigned correctly, which indicates checkpoint dir and checkpoint file name respectively。

We provide instructions for FastNN model libraty on "Local Trial" and "PAI Trial" as following.

#### 2.2.1 Local Trial

FastNN for now does not support PAISoar locally, wrt. no support for local distribution. If only for single-gpu training task, user param 'enable_paisoar' should set False, and software requirements lists:

|software|version|
| :-----: | :----: |
|python|>=2.7.6|
|TensorFlow|>=1.8|
|CUDA|>= 9.0|
|cuDNN| >= 7.0|

We take training task of Resnet-v1-50 on cifar10 for example to clarify local trial mannual。
##### 2.2.1.1 Pretrain Shell Script

```
DATASET_DIR=/tmp/data/cifar10
TRAIN_FILES=cifar10_train.tfrecord
python train_image_classifiers.py \
	--task_type=pretrain \ 
	--enable_paisoar=False \
	--dataset_name=cifar10 \
	--train_files="${TRAIN_FILES}" \
	--dataset_dir="${DATASET_DIR}" \
	--model_name=resnet_v1_50
```
##### 2.2.1.2 Finetune Shell Script

```
MODEL_DIR=/path/to/model_ckpt
CKPT_FILE_NAME=resnet_v1_50.ckpt
DATASET_DIR=/tmp/data/cifar10
TRAIN_FILES=cifar10_train.tfrecord
python train_image_classifiers.py \
	--task_type=finetune \
	--enable_paisoar=False \
	--dataset_name=cifar10 \
	--train_files="${TRAIN_FILES}" \
	--dataset_dir="${DATASET_DIR}" \
	--model_name=resnet_v1_50 \
	--model_dir="${MODEL_DIR}" \
	--ckpt_file_name="${CKPT_FILE_NAME}"
```

#### 2.2.2 PAI Trial
PAI supports several state-of-art frameworks, including TensorFlow(compatible with community version of 1.4 and 1.8), MXNet(0.9.5), Caffe(rc3). However, only tensorflow applies built-in PAISoar and support distributed training of multi-gpus and multi-nodes. For mannual, please turn to [FastNN-On-PAI](https://yuque.antfin-inc.com/docs/share/1368e10c-45f1-443e-88aa-0bb5425fea72)。

## 3. User Parameters Intructions
Chapter 2.2 explains some most import params, While still many params stay unknown to users. FastNN model library integrates requirements form models and PAISoar, summarizes params in file 'flags.py'(which also allows new params self-defined), which can be divided into 6 parts:

* Dataset Option: Specific training dataset, such as 'dataset_dir' indicating training dataset directory;
* Dataset PreProcessing Option: Specific preprocessing func and params on dataset pipeline;
* Model Params Option: Specific model base params, including model_name、batch_size;
* Learning Rate Tuning: Specific learning rate and relative tuning params;
* Optimizer Option: Specific optimizer and relative tuning params;
* Logging Option: Specific params for logging;
* Performance Tuning: Specific half-precision and other relative tuning params.

### 3.1 Dataset Option

|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|dataset_name|string|Indicating dataset name, default mock|
|dataset_dir|string|Indicating path to input dataset, default None|
|num_sample_per_epoch|integer|Total num of samples in training dataset|
|num_classes|integer|Classes of training dataset, default 100|
|train_files|string|String of name of all training files separated by comma, such as"0.tfrecord,1.tfrecord"|

### 3.2 Dataset Preprocessing Tuning

|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|preprocessing_name|string|Preprocessing func name, defult None|
|shuffle_buffer_size|integer|shuffle buffer size of training dataset, default 1024|
|num_parallel_batches|integer|Product with batch_size indicates value of map_and_batch, default 8|
|prefetch_buffer_size|integer|Prefetch N batches data into dataset pipeline, default N 32|
|num_preprocessing_threads|integer|Number of preprocessing threads, default 16|
|datasets_use_caching|bool|Cache the compressed input data in memory. This improves the data input performance, at the cost of additional memory|

### 3.3 Model Params Option

|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|task_type|string|support pretrain or finetune, default pretrain|
|model_name|string|Indicating name of model，default inception_resnet_v2|
|num_epochs|integer|Number of training epochs，default 100|
|weight_decay|float|The weight decay on the model weights, default 0.00004|
|max_gradient_norm|float|clip gradient to this global norm, default None for clip-by-global-norm diabled|
|batch_size|integer|The number of samples in each batch, default 32|
|model_dir|string|dir of checkpoint for init|
|ckpt_file_name|string|Initial checkpoint (pre-trained model: base_dir + model.ckpt).|

### 3.4 Learning Rate Tuning

|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|warmup_steps|integer|how many steps we inverse-decay learning. default 0.|
|warmup_scheme|string|how to warmup learning rates. Options include:'t2t' refers to Tensor2Tensor way, start with lr 100 times smaller,then exponentiate until the specified lr. default 't2t'|
|decay_scheme|string|How we decay learning rate. Options include:1、luong234: after 2/3 num train steps, we start halving the learning rate for 4 times before finishing;2、luong5: after 1/2 num train steps, we start halving the learning rate for 5 times before finishing;3、luong10: after 1/2 num train steps, we start halving the learning rate for 10 times before finishing.|
|learning_rate_decay_factor|float|learning rate decay factor, default 0.94|
|learning_rate_decay_type|string|specifies how the learning rate is decayed. One of ["fixed", "exponential", or "polynomial"], default exponential|
|learning_rate|float|Starting value for learning rate, default 0.01|
|end_learning_rate|float|Lower bound for learning rate when decay not disabled, default 0.0001|

### 3.5 Optimizer Option

|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|optimizer|string|the name of the optimizer, one of "adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd" or "rmsprop". Default "rmsprop"|
|adadelta_rho|float|the decay rate for adadelta, default 0.95, specially for Adadelta|
|adagrad_initial_accumulator_value|float|starting value for the AdaGrad accumulators, default 0.1, specially for Adagrada|
|adam_beta1|float|the exponential decay rate for the 1st moment estimates, default 0.9, specially for Adam|
|adam_beta2|float|the exponential decay rate for the 2nd moment estimates, default 0.999, specially for Adam|
|opt_epsilon|float|epsilon term for the optimizer, default 1.0, specially for Adam|
|ftrl_learning_rate_power|float|the learning rate power, default -0.5, specially for Ftrl|
|ftrl_initial_accumulator_value|float|Starting value for the FTRL accumulators, default 0.1, specially for Ftrl|
|ftrl_l1|float|The FTRL l1 regularization strength, default 0.0, specially for Ftrl|
|ftrl_l2|float|The FTRL l2 regularization strength, default 0.0, specially for Ftrl|
|momentum|float|The momentum for the MomentumOptimizer, default 0.9, specially for Momentum|
|rmsprop_momentum|float|Momentum for the RMSPropOptimizer, default 0.9|
|rmsprop_decay|float|Decay term for RMSProp, default 0.9|

### 3.6 Logging Option

|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|stop_at_step|integer|the whole training steps, default 100|
|log_loss_every_n_iters|integer|frequency to print loss info, default 10|
|profile_every_n_iters|integer|frequency to print timeline, default 0|
|profile_at_task|integer|node index to output timeline, default 0|
|log_device_placement|bool|whether or not to log device placement, default False|
|print_model_statistics|bool|whether or not to print trainable variables info, default false|
|hooks|string|specify hooks for training, default "StopAtStepHook,ProfilerHook,LoggingTensorHook,CheckpointSaverHook"|

### 3.7 Performanse Tuning Option

|#Name|#Type|#Description|
| :-----: | :----: | :-----|
|use_fp16|bool|whether to train with fp16, default True|
|loss_scale|float|loss scale value for training, default 1.0|
|enable_paisoar|bool|whether or not to use pai soar，default True.|
|protocol|string|default grpc.For rdma cluster, use grpc+verbs instead|

## 4. Self-defined Model Exploration

If existing models can't meet your requirements, we allow inheriting dataset／models／preprocessing api for self-defined exploration. Before that, you may need to understand overall code architecture of FastNN model library(taking 'images' models for example, whose main func file is 'train_image_classifiers.py'):

```
# Initialize some model in models for network_fn, and may carries param 'train_image_size'
    network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=FLAGS.num_classes,
            weight_decay=FLAGS.weight_decay,
            is_training=(FLAGS.task_type in ['pretrain', 'finetune']))
# Initialize some preprocess_fn
    preprocessing_fn = preprocessing_factory.get_preprocessing(
                FLAGS.model_name or FLAGS.preprocessing_name,
                is_training=(FLAGS.task_type in ['pretrain', 'finetune']))
# According to dataset_name，choose right tfrecord for training dataset, and call preprocess_fn to parse training dataset
    dataset_iterator = dataset_factory.get_dataset_iterator(FLAGS.dataset_name,
                                                            train_image_size,
                                                            preprocessing_fn,
                                                            data_sources,
# Based on network_fn、dataset_iterator，define loss_fn
    def loss_fn():
    	with tf.device('/cpu:0'):
      		images, labels = dataset_iterator.get_next()
        logits, end_points = network_fn(images)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=tf.cast(logits, tf.float32), weights=1.0)
        if 'AuxLogits' in end_points:
          loss += tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=tf.cast(end_points['AuxLogits'], tf.float32), weights=0.4)
        return loss
# Call PAI-Soar API to wrapper loss_fn with original optimizer
    opt = paisoar.ReplicatedVarsOptimizer(optimizer, clip_norm=FLAGS.max_gradient_norm)
    loss = optimizer.compute_loss(loss_fn, loss_scale=FLAGS.loss_scale)
# Based on loss and opt, define training tensor 'train_op'
    train_op = opt.minimize(loss, global_step=global_step)
```

### 4.1 For New Dataset
**FastNN model library has supported direct access to dataset of tfrecord format**，and implements dataset pipeline based on TFRecordDataset for model training. If your datasets are some other formats, dataset pipeline should be rewritten(reference utils/dataset_utils.py). In addition, data-spliting is not finely implements, FastNN requires number of training files can be divided by number of workers and number of samples among training files is even. For example, cifar10、mnist training dataset only works in distributed jobs of multi-gpus, thus requires all samples can be allocated evenly among all workers.

If your dataset file is format of tfrecord, please reference files cifar10/mnist/flowers in 'datasets'.
* If set 'dataset_name cifar10', then create a new file 'cifar10.py' in 'datasets' as following:

```python
"""Provides data for the Cifar10 dataset.
The dataset scripts used to create the dataset can be found at:
utils/scripts/data/download_and_convert_cifar10.py
"""
from __future__ import division
from __future__ import print_function

import tensorflow as tf
"""Expect func_name is ‘parse_fn’
"""
def parse_fn(example):
  with tf.device("/cpu:0"):
    features = tf.parse_single_example(
      example,
      features={
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      }
    )
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    label = features['image/class/label']
    return image, label
```

* Supplement 'dataset_map' in file 'datasets/dataset_factory.py'
* When run scripts, user need set 'train_files'(refer to chapter 3.1)

### 4.2 For New Model
For exploration of new models, please turn to files in 'images/models':

* Model convergents normally;
* Input or output apis look like models in "models", Type or shape of input apis share with that of preprocessing and Type or shape of output apis share with that of labels;
* No other distributed settings.

Here, datasets and preprocessing can be reused. However, you gonna the following supplements:

* Supplement 'models_map' and 'arg_scopes_map' in file 'models/model_factory.py';
* Import your model into 'images/models'.

### 4.3 For New Dataset Preprocessing
For exploration of new dataset preprocessing_fn, please turn to files in 'images/preprocessing':

* Input or output apis look like preprocessing_fn in directory "preprocessing", Type or shape of input apis share with that of dataset and Type or shape of output apis share with that of model inputs;

you gonna the following supplements:

* Supplement 'preprocessing_fn_map' in file 'preprocessing_factory.py';

* Import yout preprocessing func file into directory 'images/preprocessing'.

### 4.4 For New Loss_fn
For 'images' in FastNN model library, as main file 'train_image_classifiers.py' shows, we implement loss_fn with 'tf.losses.sparse_softmax_cross_entropy'.

You can directly modify 'loss_fn' in 'train_image_classifiers.py'. However, when trying out distributed training jobs with PAISoar, 'loss_fn' returns loss only which is limited unchangable. Any other variables can be passed globally as accuracy returned in 'loss_fn' of 'train_image_classifiers.py'.

## 5. Acknowledgements
FastNN is run as a open-source project to indicate our contribution to distributed model library based on PAISoar. We believe FastNN allow researchers to most effectively explore various neural networks, and support faster data parallelism. 
We will carry on with more innovation work on distribution, including model parallelism and gradient compression and so on.

FastNN for now includes only some state-of-art models on computer vision. However, models of NLP(Bert、XLNet、GPT-2、NMT) are comming soon. 

FastNN now focuses on data parallelism, all models in 'images/models' are noted from [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim#tensorflow-slim-image-classification-model-library).
Thanks to TensorFlow community for implementation of these models. If any questions, please email me whenever you would like.
