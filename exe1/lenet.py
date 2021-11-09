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

#运行信息
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
context.set_context(device_id=0)

#数据处理
def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1, training=True):
    # 定义数据集
    cifar10_ds = ds.Cifar10Dataset(data_path)
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 数据增强
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # 使用map映射函数，将数据操作应用到数据集
    cifar10_ds = cifar10_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    cifar10_ds = cifar10_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    cifar10_ds = cifar10_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    cifar10_ds = cifar10_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    cifar10_ds = cifar10_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    ds.config.set_seed(58)
    # 进行shuffle、batch操作
    buffer_size = 10000
    # 随机打乱数据顺序
    cifar10_ds = cifar10_ds.shuffle(buffer_size=buffer_size)
    # 对数据集进行分批
    cifar10_ds = cifar10_ds.batch(batch_size, drop_remainder=True)

    return cifar10_ds

class LeNet5(nn.Cell):
    """
    Lenet网络结构
    """
    def __init__(self, num_class=10, num_channel=3):
        super(LeNet5, self).__init__()
        # 定义所需要的运算
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class StepLossAccInfo(Callback):
    def __init__(self, model, eval_dataset, steps_loss, steps_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.steps_loss = steps_loss
        self.steps_eval = steps_eval

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_step = (cur_epoch-1)*1562 + cb_params.cur_step_num
        self.steps_loss["loss_value"].append(str(cb_params.net_outputs))
        self.steps_loss["step"].append(str(cur_step))
        if cur_step % 1562 == 0:
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False) 
            print("============== Accuracy:{} ==============".format(acc))
            self.steps_eval["step"].append(cur_step)
            self.steps_eval["acc"].append(acc["Accuracy"])

def train_net(model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """定义训练的方法"""
    # 加载训练数据集
    ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size, training=True)
    ds_test = create_dataset(os.path.join(data_path, "test"), 32, training=False)
    #打印信息
    steps_loss = {"step": [], "loss_value": []}
    steps_eval = {"step": [], "acc": []}
    step_loss_acc_info = StepLossAccInfo(model , ds_test, steps_loss, steps_eval)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125),step_loss_acc_info], dataset_sink_mode=sink_mode)
    acc = model.eval(ds_test, dataset_sink_mode=False)
    print("{}".format(acc))

if __name__ == '__main__':
    #网络实例化
    net = LeNet5()
    # 定义损失函数
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    #定义优化器
    lr = 0.01
    momentum = 0.9
    net_opt = nn.Momentum(net.trainable_params(), learning_rate=lr, momentum=momentum)

    train_epoch = 30
    dataset_size = 1
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    
    #数据处理
    cifar10_path = "./datasets/cifar-10-batches-bin"
    model_path = "./models"

    repeat_size = 1
    
    os.system('rm -f {0}*.ckpt {0}*.meta {0}*.pb'.format(model_path))
    # 设置模型保存参数
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    # 应用模型保存参数
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet",directory=model_path, config=config_ck)

    train_net(model, train_epoch, cifar10_path, dataset_size, ckpoint, False)