import torch
import torch.nn as nn
from torchsummary import summary


class Program():
    def __init__(self):
        pass

class ResNet50(nn.Module):
    def __init__(self, out_activations, in_channels=3):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, stride=2, kernel_size=7, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bottleneck_layers = [BottleneckBlock(64,64)] + [BottleneckBlock(256, 64) for _ in range(2)]
        self.bottleneck_layers += [BottleneckBlock(256,128,stride=2)] + [BottleneckBlock(512,128) for _ in range(3)]
        self.bottleneck_layers += [BottleneckBlock(512, 256,stride=2)] + [BottleneckBlock(1024, 256) for _ in range(5)]
        self.bottleneck_layers += [BottleneckBlock(1024, 512,stride=2)] + [BottleneckBlock(2048, 512) for _ in range(2)]
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fully_connected = nn.Linear(2048, out_activations)
    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)

        for layer in self.bottleneck_layers:
            x = layer(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.fully_connected(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, downsampled_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        self.c_projection = nn.Conv2d(in_channels, downsampled_channels*4, kernel_size=1, stride=stride)
        self.activation = nn.ReLU()

        self.c1 = nn.Conv2d(in_channels, downsampled_channels, kernel_size=1, stride=stride)
        self.c1_batchnorm = nn.BatchNorm2d(downsampled_channels)

        self.c2 = nn.Conv2d(downsampled_channels, downsampled_channels, kernel_size=3, padding=1)
        self.c2_batchnorm = nn.BatchNorm2d(downsampled_channels)

        self.c3 = nn.Conv2d(downsampled_channels, downsampled_channels*4, kernel_size=1)
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

dummy_data = torch.rand(size=(10,1,224,224))
dummy_model = ResNet50(20, in_channels=1)
print(dummy_model(dummy_data).shape)
# print(summary(ResNet50(1), input_size=(1, 224, 224)))
        