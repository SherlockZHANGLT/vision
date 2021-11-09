#运行信息
import os
import argparse
from mindspore import context
#数据处理
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
#数据增强
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision.c_transforms as c_vision
#创建模型
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
#训练及保存模型
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
# 导入模型训练需要的库
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model

from mindspore.train.callback import Callback

conv2d=nn.MaxPool2d(kernel_size=2, stride=2)
input_x = Tensor(np.ones([1, 32, 32, 32]), mindspore.float32)

print(conv2d(input_x).shape)