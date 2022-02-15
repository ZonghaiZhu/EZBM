# coding:utf-8
'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44',
           'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class LambdaLayer(nn.Module):
    def __init__(self, planes):
        super(LambdaLayer, self).__init__()
        self.planes = planes

    def forward(self, x):
        # return self.lambd(x)
        # # ::2 步长为2到结束, x[:, :, ::2, ::2]->128,16,16,16,
        # 然后对channel前后各填充self.planes//4=8,使得channel从16->32
        # 相当于x从128,16,32,32->128,32,16,16
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes // 4, self.planes // 4), "constant", 0)


class BasicBlock(nn.Module):
    expansion = 1
    # 对于每一个block，其in_planes, planes都相等, 第二个stride=2, 且in_planes=16, planes=32, 要shortcut
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) # k=3, s=1, p=1是结构不变的, s=2则缩一半
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A. 
                """
                # input：四维或者五维的tensor Variabe
                # pad：不同Tensor的填充方式
                #     1.四维Tensor：传入四元素tuple(pad_l, pad_r, pad_t, pad_b)，
                #     指的是（左填充，右填充，上填充，下填充），其数值代表填充次数
                #     2.六维Tensor：传入六元素tuple(pleft, pright, ptop, pbottom, pfront, pback)，
                #     指的是（左填充，右填充，上填充，下填充，前填充，后填充），其数值代表填充次数
                # mode： ’constant‘, ‘reflect’ or ‘replicate’三种模式，指的是常量，反射，复制三种模式
                # value：填充的数值，在"contant"模式下默认填充0，mode="reflect" or "replicate"时没有
                #     value参数
                # self.shortcut = LambdaLayer(lambda x:
                #                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant"0))
                self.shortcut = LambdaLayer(planes)

            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # 对于layer2 x:128,16,32,32->128,32,16,16
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False):
        super(ResNet_s, self).__init__()
        self.in_planes = 16
        # conv2d->(28-kernel_size+2*padding)/stride+1, 向下取整: 128,3,32,32->128,16,32,32
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16) # num_blocks = [5,5,5]
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if use_norm:
            self.linear = NormedLinear(64, num_classes)
        else:
            self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1) # num_blocks[0] = 5, strides->[1,1,1,1,1]
        layers = [] # layer2 num_blocks[1] = 5, strides->[2,1,1,1,1], 第一步stride=2
        for stride in strides: # layer2, 第一步self.in_planes=16, planes=31, 第一步走完后self.in_planes=32
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # x:128,3,32,32变成->128,16,32,32
        out = self.layer1(out) # layer1 第一层输入输出channels都是16
        out = self.layer2(out) # layer2 size:32x32->16x16 channel:16->32 进入第一步stride=2
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10, use_norm=False):
    return ResNet_s(BasicBlock, [3, 3, 3], num_classes=num_classes, use_norm=use_norm)


def resnet32(num_classes=10, use_norm=False):
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])
