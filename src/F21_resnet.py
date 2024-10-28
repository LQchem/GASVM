import numpy as np
import torch
import pandas
import torchvision
import torch.optim as optim

import matplotlib.pyplot as plt
from F01_loaddata import *
from F02_parameters import *
from F11_Generator_reformat import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

        self.activation_func = self.generator.activation_func1
    def forward(self, x):
#        print("forward in basic block")       

  #      out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.activation_func((self.bn1(self.conv1(x))))
 

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out  = self.activation_func(out)
  #      out = nn.ReLU()(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, generator):
        super(ResNet18, self).__init__()
        self.in_channels = 1
        out_channel = 8  #32 nees 7.6h
        self.conv1 = nn.Conv2d(1, out_channel, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channel)
        

        num_blocks = [2,2,2,2]
        stride_list = [1,2,2,2] #do not change easily, or errors



        # 定义 ResNet 的各个阶段
        layers = []
        layers.extend(self._make_layer(generator,BasicBlock, out_channel*4,  num_blocks[0],  stride=stride_list[0]))
        layers.extend(self._make_layer(generator,BasicBlock, out_channel*4,  num_blocks[1],  stride=stride_list[1]))
        layers.extend(self._make_layer(generator,BasicBlock, out_channel*4,  num_blocks[2],  stride=stride_list[2]))
        layers.extend(self._make_layer(generator,BasicBlock, out_channel*8, num_blocks[3] , stride=stride_list[3]))

        # 动态计算特征图的最终尺寸
        final_size = self._calculate_final_size(input_size=(28, 28))
        
        # 设置平均池化的大小
        layers.append(nn.AvgPool2d(final_size))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(out_channel*8 * BasicBlock.expansion, generator.out_Nfeature))

        self.layers = nn.Sequential(*layers)

    def _make_layer(self, generator,block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
	
	#---lq add---
        activation_func = generator.activation_func1
        Normalizer      = generator.Normalizer
        block.generator = generator
	#--end-----

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            
            #===lq add===
       #     layers.append(activation_func)
       #     layers.append(Normalizer(out_channels))
            #====end===

            self.in_channels = out_channels * block.expansion
        return layers

    def _calculate_final_size(self, input_size):
        """
        计算经过多次下采样后的特征图的最终尺寸。
        """
        h, w = input_size
        for stride in [1, 2, 2, 2]:
            h = h // stride
            w = w // stride
        return h, w

    def forward(self, x):
        return self.layers(x)


if __name__=="__main__":
	g_p, args = f02_main()
	

	generator = Generator(g_p)


	# 创建 ResNet-18 模型实例
	resnet_model = ResNet18(generator)

#	print(resnet_model.__dict__)
	print(resnet_model)
##	print(generator)

	


