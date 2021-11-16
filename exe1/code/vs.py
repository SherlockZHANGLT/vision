import os
import torch
import cv2
import numpy as np
import torch.optim
import torchvision
from PIL import Image
import os
import shutil
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 30
LR = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 32
BASE_CHANNEL = 32
INPUT_CHANNEL = 1
INPUT_SIZE = 28
pthfile = './save_model/epoch_20.pth'
IMAGE_FOLDER = './save_fim'
INSTANCE_FOLDER = None

feature_result = None
sz=32

def hook_func(layer, data_input, data_output):
    global feature_result
    feature_result = data_output

def autoNorm(data):         #传入一个矩阵
    mins = data.min(0)      #返回data矩阵中每一列中最小的元素，返回一个列表
    maxs = data.max(0)      #返回data矩阵中每一列中最大的元素，返回一个列表
    ranges = maxs - mins    #最大值列表 - 最小值列表 = 差值列表
    normData = np.zeros(np.shape(data))     #生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据
    row = data.shape[0]                     #返回 data矩阵的行数
    normData = data - np.tile(mins,(row,1)) #data矩阵每一列数据都减去每一列的最小值
    normData = normData / np.tile(ranges,(row,1))   #data矩阵每一列数据都除去每一列的差值（差值 = 某列的最大值- 某列最小值）
    return normData

class Model(nn.Module):
    def __init__(self, input_ch, num_classes, base_ch):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.base_ch = base_ch
        self.feature_length = 1600

        self.net = nn.Sequential(
            nn.Conv2d(input_ch, base_ch, kernel_size=3),
            nn.BatchNorm2d(base_ch, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, kernel_size=3),
            nn.BatchNorm2d(base_ch, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_ch, base_ch*2, kernel_size=3),
            nn.BatchNorm2d(base_ch*2, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Conv2d(base_ch*2, base_ch*2, kernel_size=3),
            nn.BatchNorm2d(base_ch*2, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(1600,400),
            nn.PReLU(),
            nn.Linear(400,120),
            nn.PReLU(),
            nn.Linear(120,10)
        )

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, self.feature_length)
        output = self.fc(output)
        return output

if __name__ == '__main__':
    model=Model(input_ch=3, num_classes=10, base_ch=BASE_CHANNEL).cuda()
    model.load_state_dict(torch.load(pthfile))
    model.eval()
    print(model.net[4])
    model.net[10].register_forward_hook(hook_func)

    data = np.uint8(np.random.uniform(150, 180, (32, 32, 3)))/255
    for i in range(20):
        data = torch.tensor(data.transpose((2, 0, 1))).unsqueeze(0).to(torch.float32)      
        data= data.cuda()  
        data.requires_grad = True                             
        optim = torch.optim.Adam([data], lr=0.1, weight_decay=1e-6)  
 
        for n in range(20):
            optim.zero_grad()
            model(data)
            loss = -1 * feature_result[0, 4].mean()
            loss.backward() 
            optim.step()
            print(f'epoch:{i}, level:{n}, loss: {loss.item()}, data-mean:{data[0].mean()}')
 
        data = data.data.cpu().numpy()[0].transpose(1, 2, 0)
    for i in range(3):
        data[:,:,i]=autoNorm(data[:,:,i])
    cv2.imwrite('ans.png', data)