#运行信息
import os
import argparse
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
#训练及保存模型
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
# 导入模型训练需要的库
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor

from mindspore.train.callback import Callback
import stat
from mindspore import Model, Tensor, context, save_checkpoint, load_checkpoint, load_param_into_net

import matplotlib.pyplot as plt

#运行信息
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
context.set_context(device_id=0)

cifar10_path = "./datasets/cifar-10-batches-bin"
train_data_path = os.path.join(cifar10_path, "train")
val_data_path = os.path.join(cifar10_path, "test")
model_name='./model/best.ckpt'
imsize=32

def autoNorm(data):         #传入一个矩阵
    mins = data.min(0)      #返回data矩阵中每一列中最小的元素，返回一个列表
    maxs = data.max(0)      #返回data矩阵中每一列中最大的元素，返回一个列表
    ranges = maxs - mins    #最大值列表 - 最小值列表 = 差值列表
    normData = np.zeros(np.shape(data))     #生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据
    row = data.shape[0]                     #返回 data矩阵的行数
    normData = data - np.tile(mins,(row,1)) #data矩阵每一列数据都减去每一列的最小值
    normData = normData / np.tile(ranges,(row,1))   #data矩阵每一列数据都除去每一列的差值（差值 = 某列的最大值- 某列最小值）
    return normData

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
        self.conv1 = nn.Conv2d(num_channel, 32, kernel_size=3, stride=1, pad_mode="same", weight_init='XavierUniform')
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, pad_mode="same", weight_init='XavierUniform')
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, pad_mode="same", weight_init='XavierUniform')
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, pad_mode="same", weight_init='XavierUniform')
        self.fc1 = nn.Dense(64*8*8, 1024)
        self.fc2 = nn.Dense(1024, 120)
        self.fc3 = nn.Dense(120, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm2d(32, momentum=0.99, eps=0.00001, gamma_init='Uniform',
                        beta_init=Tensor(np.zeros(32).astype(np.float32)), moving_mean_init=Tensor(np.zeros(32).astype(np.float32)), moving_var_init=Tensor(np.ones(32).astype(np.float32)))
        self.bn2 = nn.BatchNorm2d(32, momentum=0.99, eps=0.00001, gamma_init='Uniform',
                        beta_init=Tensor(np.zeros(32).astype(np.float32)), moving_mean_init=Tensor(np.zeros(32).astype(np.float32)), moving_var_init=Tensor(np.ones(32).astype(np.float32)))
        self.bn3 = nn.BatchNorm2d(64, momentum=0.99, eps=0.00001, gamma_init='Uniform',
                        beta_init=Tensor(np.zeros(64).astype(np.float32)), moving_mean_init=Tensor(np.zeros(64).astype(np.float32)), moving_var_init=Tensor(np.ones(64).astype(np.float32)))
        self.bn4 = nn.BatchNorm2d(64, momentum=0.99, eps=0.00001, gamma_init='Uniform',
                        beta_init=Tensor(np.zeros(64).astype(np.float32)), moving_mean_init=Tensor(np.zeros(64).astype(np.float32)), moving_var_init=Tensor(np.ones(64).astype(np.float32)))
        self.dropout=nn.Dropout(0.75)

    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x=self.bn1(x)
        x = self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x=self.dropout(x)
        x = self.conv3(x)
        x=self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x=self.bn4(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x=self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def visualize_model(best_ckpt_path,val_ds):
    # 定义网络并加载参数，对验证集进行预测
    net = LeNet5()
    param_dict = load_checkpoint(best_ckpt_path)
    load_param_into_net(net, param_dict)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction='mean')
    model = Model(net, loss,metrics={"Accuracy":nn.Accuracy()})
    data = next(val_ds.create_dict_iterator())
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()
    with open(cifar10_path+"/batches.meta.txt","r",encoding="utf-8") as f:
        class_name = [name.replace("\n","") for name in f.readlines()]
    output = model.predict(Tensor(data['image']))
    pred = np.argmax(output.asnumpy(),axis=1)

    # 可视化模型预测
    for i in range(len(labels)):
        plt.subplot(4,8,i+1)
        color = 'blue' if pred[i] == labels[i] else 'red'
        plt.title('{}'.format(class_name[pred[i]]), color=color)
        picture_show = np.transpose(images[i],(1,2,0))
        picture_show = picture_show/np.amax(picture_show)
        picture_show = np.clip(picture_show, 0, 1)
        plt.axis('off')
        plt.imshow(picture_show)
        plt.savefig(fname="./label/batch_"+str(i)+'.png')

    acc = model.eval(val_ds)
    print("acc:",acc)

    print(type(net.parameters_and_names()))
    for i, j in net.parameters_and_names():
        wc = j
        wc=wc.flatten()
        pic_show=np.zeros(len(wc))
        for q in range(len(wc)):
            pic_show[q]=wc[q].asnumpy()
        pic_show=np.resize(pic_show,(32,27))
        pic_show=autoNorm(pic_show)
        print(pic_show)
        plt.matshow(pic_show, cmap=plt.cm.gray)
        plt.title(i)
        plt.colorbar()
        plt.savefig(fname="./conv/"+i+'.png')
        break

if __name__ == '__main__':
    val_ds = create_dataset(val_data_path,training=False)
    visualize_model(model_name, val_ds)