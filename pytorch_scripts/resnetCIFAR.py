###########################################################################
# ## ADAPTED FROM: https://github.com/akamaster/pytorch_resnet_cifar10 ####
###########################################################################
"""
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
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
"""
import csv

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from hg_noise_injector.hans_gruber import HansGruberNI

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B', norm='batch'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(planes // 4, planes)
            self.bn2 = nn.GroupNorm(planes // 4, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU6()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # This layer throws error in multi_gpu setting, set to B
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                if norm == 'batch':
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(self.expansion * planes)
                    )
                elif norm == 'group':
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                        nn.GroupNorm(self.expansion * planes//4, self.expansion * planes)
                    )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, inject_p=0.1, inject_epoch=0, norm='batch'):
        """ Class that represents the ResNet model
        :param block:
        :param num_blocks:
        :param num_classes:
        :param use_noise_injection: flag that defines if the noise injection is used or not
        """
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(16)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(4, 16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, norm=norm)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, norm=norm)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, norm=norm)
        self.linear = nn.Linear(64, num_classes)
        self.relu = nn.ReLU6()

        self.noise_injector = HansGruberNI(p=inject_p, inject_epoch=inject_epoch)
        self.apply(_weights_init)

    def load_noise_file(self, noise_file_path):
        with open(noise_file_path) as fp:
            noise_data = list(csv.DictReader(fp))
        self.noise_injector.set_noise_data(noise_data)

    def _make_layer(self, block, planes, num_blocks, stride, norm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm=norm))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, inject=True, current_epoch=0):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.noise_injector(out, inject, current_epoch)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(n_classes=10, inject_p=0.1, inject_epoch=0, norm='batch'):
    return ResNet(BasicBlock, [3, 3, 3], n_classes, inject_p, inject_epoch, norm)


def resnet32(n_classes=10, inject_p=0.1, inject_epoch=0, norm='batch'):
    return ResNet(BasicBlock, [5, 5, 5], n_classes, inject_p, inject_epoch, norm)


def resnet44(n_classes=10, inject_p=0.1, inject_epoch=0, norm='batch'):
    return ResNet(BasicBlock, [7, 7, 7], n_classes, inject_p, inject_epoch, norm)


def resnet56(n_classes=10, inject_p=0.1, inject_epoch=0, norm='batch'):
    return ResNet(BasicBlock, [9, 9, 9], n_classes, inject_p, inject_epoch, norm)


def resnet110(n_classes=10, inject_p=0.1, inject_epoch=0, norm='batch'):
    return ResNet(BasicBlock, [18, 18, 18], n_classes, inject_p, inject_epoch, norm)


def resnet1202(n_classes=10, inject_p=0.1, inject_epoch=0, norm='batch'):
    return ResNet(BasicBlock, [200, 200, 200], n_classes, inject_p, inject_epoch, norm)
