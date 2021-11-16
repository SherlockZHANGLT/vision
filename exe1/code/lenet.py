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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
context.set_context(device_id=0)

train_epoch = 50

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

# 模型验证
def apply_eval(eval_param):
    eval_model = eval_param['model']
    eval_ds = eval_param['dataset']
    metrics_name = eval_param['metrics_name']
    res = eval_model.eval(eval_ds)
    return res[metrics_name]

class LeNet5(nn.Cell):
    """
    Lenet网络结构
    """
    def __init__(self, num_class=10, num_channel=3):
        super(LeNet5, self).__init__()
        # 定义所需要的运算
        self.conv1 = nn.Conv2d(num_channel, 32, kernel_size=3, stride=1,weight_init='XavierUniform')
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1,  weight_init='XavierUniform')
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1,  weight_init='XavierUniform')
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, weight_init='XavierUniform')
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

class EvalCallBack(Callback):
    def __init__(self, eval_function, eval_param_dict,model, eval_dataset, steps_loss, steps_eval,interval=1,eval_start_epoch=1,save_best_ckpt=True,ckpt_directory="./model/", besk_ckpt_name="best.ckpt", metrics_name="acc"):
        self.eval_param_dict = eval_param_dict
        self.eval_function = eval_function
        self.model = model
        self.eval_dataset = eval_dataset
        self.steps_loss = steps_loss
        self.steps_eval = steps_eval
        self.eval_start_epoch = eval_start_epoch
        self.interval = interval
        self.save_best_ckpt = save_best_ckpt
        self.best_res = 0
        self.best_epoch = 0
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.best_ckpt_path = os.path.join(ckpt_directory, besk_ckpt_name)
        self.metrics_name = metrics_name
        self.loss_d=[]
        self.res_d=[]

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_step = (cur_epoch-1)*1562 + cb_params.cur_step_num
        self.steps_loss["loss_value"].append(str(cb_params.net_outputs))
        self.steps_loss["step"].append(str(cur_step))

        # 删除ckpt文件
    def remove_ckpoint_file(self, file_name):
        os.chmod(file_name, stat.S_IWRITE)
        os.remove(file_name)

    # 每一个epoch后，打印训练集的损失值和验证集的模型精度，并保存精度最好的ckpt文件
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        loss_epoch = cb_params.net_outputs
        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            res = self.eval_function(self.eval_param_dict)
            print('Epoch {}/{}'.format(cur_epoch, train_epoch))
            print('-' * 10)
            print('train Loss: {}'.format(loss_epoch))
            print('val Acc: {}'.format(res))
            if res >= self.best_res:
                self.best_res = res
                self.best_epoch = cur_epoch
                if self.save_best_ckpt:
                    if os.path.exists(self.best_ckpt_path):
                        self.remove_ckpoint_file(self.best_ckpt_path)
                    save_checkpoint(cb_params.train_network, self.best_ckpt_path)
            self.loss_d.append(loss_epoch.asnumpy())
            self.res_d.append(res)

    # 训练结束后，打印最好的精度和对应的epoch
    def end(self, run_context):
        print("End training, the best {0} is: {1}, the best {0} epoch is {2}".format(self.metrics_name,
                                                                                     self.best_res,
                                                                                     self.best_epoch), flush=True)
        x=np.arange(1,train_epoch+1)
        _,ax1=plt.subplots()
        ax2=ax1.twinx()
        ax1.plot(x,self.loss_d, color="red",label='train loss')
        ax2.plot(x,self.res_d, color="blue",label='val Acc')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('train loss')
        ax2.set_ylabel('val Acc')
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        plt.savefig(fname="lenet1_trainning.png")

def train_net(model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """定义训练的方法"""
    # 加载训练数据集
    ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size, training=True)
    ds_test = create_dataset(os.path.join(data_path, "test"), 32, training=False)
    #打印信息
    steps_loss = {"step": [], "loss_value": []}
    steps_eval = {"step": [], "acc": []}
    eval_param_dict = {"model":model,"dataset":ds_test,"metrics_name":"Accuracy"}
    step_loss_acc_info = EvalCallBack(apply_eval, eval_param_dict,model , ds_test, steps_loss, steps_eval)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125),step_loss_acc_info], dataset_sink_mode=sink_mode)

if __name__ == '__main__':
    #网络实例化
    net = LeNet5()
    # 定义损失函数
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    #定义优化器
    lr = 0.01
    momentum = 0.9
    net_opt = nn.Momentum(net.trainable_params(), learning_rate=lr, momentum=momentum)

    dataset_size = 1
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    
    #数据处理
    cifar10_path = "./datasets/cifar-10-batches-bin"
    model_path = "./models"
    
    # 设置模型保存参数
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    # 应用模型保存参数
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet",directory=model_path, config=config_ck)

    train_net(model, train_epoch, cifar10_path, dataset_size, ckpoint, False)