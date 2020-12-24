import torch
import torch.nn as nn



class ResNet(nn.Module):
    def __init__(self, out_activations, in_channels=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, stride=2, kernel_size=7, padding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.fully_connected = nn.Linear(2048, out_activations)
    
    def make_bottleneck_layer(self, blocks_number, in_channels, downsampling_factor, stride):
        stacked_bottlenecks = []
        downsampled_block = BottleneckBlock(in_channels=in_channels, stride=stride, downsampling_factor=downsampling_factor)
        in_channels = in_channels//downsampling_factor*4
        stacked_bottlenecks.append(downsampled_block)
        for num in range(blocks_number - 1):
            normal_block = BottleneckBlock(in_channels=in_channels, stride=1, downsampling_factor=4)
            stacked_bottlenecks.append(normal_block)
        return nn.Sequential(*stacked_bottlenecks)
    
    def make_basic_layer(self, blocks_number, in_channels, expansion, stride):
        stacked_bottlenecks = []
        for num in range(blocks_number - 1):
            normal_block = BasicBlock(in_channels=in_channels, stride=1, expansion=1)
            stacked_bottlenecks.append(normal_block)
        expansion_block = BasicBlock(in_channels, expansion=expansion, stride=stride)
        stacked_bottlenecks.append(expansion_block)
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

class BasicBlock(nn.Module):
    def __init__(self, in_channels, expansion, stride=1):
        super(BasicBlock, self).__init__()
        expanded_channels = int(in_channels * expansion)

        self.c_projection = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=stride, bias=False)
        self.activation = nn.ReLU()
        
        self.c1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, bias=False)
        self.c1_batchnorm = nn.BatchNorm2d(in_channels)

        self.c2 = nn.Conv2d(in_channels, expanded_channels, kernel_size=3, padding=1, bias=False)
        self.c2_batchnorm = nn.BatchNorm2d(expanded_channels)

    def forward(self, x):
        identity = x
        x = self.c1(x)
        x = self.c1_batchnorm(x)
        x = self.activation(x)

        x = self.c2(x)
        x = self.c2_batchnorm(x)
        identity = self.c_projection(identity)
        x += identity
        x = self.activation(x)
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

class ResNet18(ResNet):
    def __init__(self, out_activations, in_channels=1):
        super(ResNet18, self).__init__(out_activations=out_activations, in_channels=in_channels)
        self.layer1 = nn.Sequential(self.make_basic_layer(2, 64, 2, 1))
        self.layer2 = nn.Sequential(self.make_basic_layer(2, 128, 2, 2))
        self.layer3 = nn.Sequential(self.make_basic_layer(2, 256, 2, 2))
        self.layer4 = nn.Sequential(self.make_basic_layer(2, 512, 2, 2))

class ResNet34(ResNet):
    def __init__(self, out_activations, in_channels=1):
        super(ResNet34, self).__init__(out_activations=out_activations, in_channels=in_channels)
        self.layer1 = nn.Sequential(self.make_basic_layer(3, 64, 2, 1))
        self.layer2 = nn.Sequential(self.make_basic_layer(4, 128, 2, 2))
        self.layer3 = nn.Sequential(self.make_basic_layer(6, 256, 2, 2))
        self.layer4 = nn.Sequential(self.make_basic_layer(3, 512, 2, 2))

class ResNet50(ResNet):
    def __init__(self, out_activations, in_channels=1):
        super(ResNet50, self).__init__(out_activations=out_activations, in_channels=in_channels)
        self.layer1 = nn.Sequential(self.make_bottleneck_layer(3, 64, 1, 1))
        self.layer2 = nn.Sequential(self.make_bottleneck_layer(4, 256, 2, 2))
        self.layer3 = nn.Sequential(self.make_bottleneck_layer(6, 512, 2, 2))
        self.layer4 = nn.Sequential(self.make_bottleneck_layer(3, 1024, 2, 2))

class ResNet101(ResNet):
    def __init__(self, out_activations, in_channels=1):
        super(ResNet101, self).__init__(out_activations=out_activations, in_channels=in_channels)
        self.layer1 = nn.Sequential(self.make_bottleneck_layer(3, 64, 1, 1))
        self.layer2 = nn.Sequential(self.make_bottleneck_layer(4, 256, 2, 2))
        self.layer3 = nn.Sequential(self.make_bottleneck_layer(23, 512, 2, 2))
        self.layer4 = nn.Sequential(self.make_bottleneck_layer(3, 1024, 2, 2))

class ResNet152(ResNet):
    def __init__(self, out_activations):
        super(ResNet152, self).__init__(out_activations=out_activations)
        self.layer1 = nn.Sequential(self.make_bottleneck_layer(3, 64, 1, 1))
        self.layer2 = nn.Sequential(self.make_bottleneck_layer(4, 256, 2, 2))
        self.layer3 = nn.Sequential(self.make_bottleneck_layer(36, 512, 2, 2))
        self.layer4 = nn.Sequential(self.make_bottleneck_layer(3, 1024, 2, 2))
