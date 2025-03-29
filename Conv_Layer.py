import torch
import torch.nn as nn
from Pyramid_Dyconv import Dy_scale_Conv
from Pyramid_Dyconv import Scale_Conv
from Pyramid_Dyconv import Dy_Conv

"""""
class ConvLayer2(nn.Module):
    def __init__(self,outchannel):
        super(ConvLayer2, self).__init__()
        #self.dy_conv1 = Dy_scale_Conv(num_levels=3,inchannels=3,outchannels=outchannel)
        #self.dy_conv2 = Dy_scale_Conv(num_levels=3,inchannels=outchannel,outchannels=outchannel)
        self.dy_conv1 = Dy_scale_Conv(num_levels=3,inchannels=3,outchannels=128)
        self.bn = nn.BatchNorm2d(128)
        self.dy_conv2 = Dy_scale_Conv(num_levels=3,inchannels=128,outchannels=outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.bn(self.dy_conv1(x)))
        x = self.pooling(x)
        x = self.relu(self.dy_conv2(x))
        x = self.pooling(x)
        return x
"""""
   
class ConvLayer3(nn.Module):
    def __init__(self,outchannel):
        super(ConvLayer3, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,256,3,1,1)
        self.bn2 = nn.BatchNorm2d(256)
        self.dy_conv = Dy_scale_Conv(num_levels=3,inchannels=256,outchannels=outchannel)
        self.bn3 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dy_conv(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x
"""""
class ConvLayer4(nn.Module):
    def __init__(self,outchannel):
        super(ConvLayer4, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,256,3,1,1)
        self.bn2 = nn.BatchNorm2d(256)
        self.dy_conv = Scale_Conv(num_levels=3,inchannels=256,outchannels=outchannel)
        self.bn3 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dy_conv(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x


class ConvLayer5(nn.Module):
    def __init__(self,outchannel):
        super(ConvLayer5, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,256,3,1,1)
        self.bn2 = nn.BatchNorm2d(256)
        self.dy_conv = Dy_Conv(num_levels=3,inchannels=256,outchannels=outchannel)
        self.bn3 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dy_conv(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x
"""""

"""""
input = torch.randn(1,3,32,32)
model = ConvLayer2(16)
output = model(input)
print(output.shape)
# 计算可学习参数量的函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# 计算并打印可学习参数量
num_params = count_parameters(model)
print(f"可学习参数量: {num_params}")
"""""