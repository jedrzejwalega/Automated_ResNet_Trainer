import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.transforms import Resize
import numpy as np
import random

class ResNet50(nn.Module):
    def __init__(self, out_activations, in_channels=1):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, stride=2, kernel_size=7, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(self.__make_layer(3, 64, 1, 1))
        self.layer2 = nn.Sequential(self.__make_layer(4, 256, 2, 2))
        self.layer3 = nn.Sequential(self.__make_layer(6, 512, 2, 2))
        self.layer4 = nn.Sequential(self.__make_layer(3, 1024, 2, 2))
        self.fully_connected = nn.Linear(2048, out_activations)
    
    def __make_layer(self, blocks_number, in_channels, downsampling_factor, stride):
        stacked_bottlenecks = []
        downsampled_block = BottleneckBlock(in_channels=in_channels, stride=stride, downsampling_factor=downsampling_factor)
        in_channels = in_channels//downsampling_factor*4
        stacked_bottlenecks.append(downsampled_block)
        for num in range(blocks_number - 1):
            normal_block = BottleneckBlock(in_channels=in_channels, stride=1, downsampling_factor=4)
            stacked_bottlenecks.append(normal_block)
        return nn.Sequential(*stacked_bottlenecks)

    def forward(self, x):
        # First conv
        x = self.conv1(x)
        # Max pooling
        x = self.max_pool(x)
        # 64-channel residual blocks
        x = self.layer1(x)
        # 256-channel residual blocks
        x = self.layer2(x)
        # 512-channel residual blocks
        x = self.layer3(x)
        # 1024-channel residual blocks
        x = self.layer4(x)
        # Global average pooling
        x = x.mean([2,3])
        # Fully connected layer
        x = self.fully_connected(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, downsampling_factor, stride=1):
        super(BottleneckBlock, self).__init__()
        downsampled_channels = in_channels // downsampling_factor

        self.c_projection = nn.Conv2d(in_channels, downsampled_channels*4, kernel_size=1, stride=stride, bias=False)
        self.activation = nn.ReLU()
        
        self.c1 = nn.Conv2d(in_channels, downsampled_channels, kernel_size=1, stride=stride, bias=False)
        self.c1_batchnorm = nn.BatchNorm2d(downsampled_channels)

        self.c2 = nn.Conv2d(downsampled_channels, downsampled_channels, kernel_size=3, padding=1, bias=False)
        self.c2_batchnorm = nn.BatchNorm2d(downsampled_channels)

        self.c3 = nn.Conv2d(downsampled_channels, downsampled_channels*4, kernel_size=1, bias=False)
        self.c3_batchnorm = nn.BatchNorm2d(downsampled_channels*4)
    
    def forward(self, x):
        identity = x
        x = self.c1(x)
        x = self.c1_batchnorm(x)
        x = self.activation(x)
        
        x = self.c2(x)
        x = self.c2_batchnorm(x)
        x = self.activation(x)

        x = self.c3(x)
        x = self.c3_batchnorm(x)
        identity = self.c_projection(identity)
        x += identity
        x = self.activation(x)
        return x
