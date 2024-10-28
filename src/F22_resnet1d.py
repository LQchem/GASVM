import torch
import torch.nn as nn
import torch.nn.functional as F

from F01_loaddata import *
from F02_parameters import *
from F11_Generator_reformat import *

def load_para(class1,class2):
	for attr_name, attr_value in vars(class2).items():
		setattr(class1,attr_name,attr_value)
	
class Bottleneck(nn.Module):
	expansion = 4  # 扩展因子，决定了输出通道数相对于中间层通道数的比例

	def __init__(self,generator, in_channels, out_channels, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		
		# 第一个1x1卷积层，减少通道数
		self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm1d(out_channels)
		
		# 中间的3x1卷积层
		self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm1d(out_channels)
		
		# 第三个1x1卷积层，恢复通道数
		self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

		self.activation_func =  generator.activation_func1#nn.ReLU(inplace=True) #original is ReLu
		self.downsample = downsample
		self.stride = stride
		
	def forward(self, x):
		identity = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.activation_func(out)
		
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.activation_func(out)
		
		out = self.conv3(out)
		out = self.bn3(out)
		
		if self.downsample is not None:
			identity = self.downsample(x)
		
		out += identity
		out = self.activation_func(out)
		
		return out

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, generator,in_channels, out_channels, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		# Convolution 1
		self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm1d(out_channels)
		
		# Convolution 2
		self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm1d(out_channels)
		
		self.downsample = downsample
		self.stride = stride
		self.activation_func = generator.activation_func1
	
	def forward(self, x):
		identity = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.activation_func(out)
		
		out = self.conv2(out)
		out = self.bn2(out)
		
		if self.downsample is not None:
			identity = self.downsample(x)
		
		out += identity  #shortcut
		out = self.activation_func(out)
		
		return out

#read in [nbatch, nchannel, nfeature]
#write out [nbatch, target_feature]
class ResNet1D(nn.Module):
	def __init__(self, generator):
		super(ResNet1D, self).__init__()
		self.args = generator.args
		self.in_channels  = generator.in_channel
		self.out_channel1 = generator.out_channel1



		#self.expansion = generator.expansion
		kernel_size = generator.kernel_size
		padding     = generator.padding
		stride      = generator.stride
		Nlayer      = generator.Nlayer		
		increment   = generator.increment1

#		num_block_list   = [3,3,3,3,3, 3,3,3,3,3]#torch.ones(Nlayer) * 3 #each member in the list defines how many blocks in that layer
#		out_channel_list = [8,8,8,8,8, 8,8,8,8,8]#torch.ones(Nlayer) *32
#		stride_list      = [1,1,1,1,1, 1,1,1,1,1]#torch.ones(Nlayer) *3	

		paraA = generator.num_block        
		paraB = generator.f22_out_channel  
		paraC = generator.f22_stride       

		num_block_list   = [paraA] * Nlayer #each member in the list defines how many blocks in that layer
		stride_list      = [paraC] * Nlayer
		#out_channel_list = [paraB] * Nlayer
		out_channel_list = []
		for i in range(Nlayer):
			o_channel = paraB + i * increment
			out_channel_list.append(o_channel)


		self.conv1 = nn.Conv1d(self.in_channels, self.out_channel1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		self.bn1  = nn.BatchNorm1d(self.out_channel1)
		self.activation_func = generator.activation_func1   #nn.ReLU(inplace=True)

		self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
		#self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

		#choose the block type
		if generator.block_type == 1:		
			block_type = BasicBlock
		else:
			block_type = Bottleneck

		self.layers = nn.ModuleList()
		for i in range(Nlayer):
			layer = self._make_layer(generator,block_type, out_channel_list[i], num_block_list[i], stride_list[i])
			self.layers.append(layer)
	

		self.avgpool = nn.AdaptiveAvgPool1d(1)

		self.fc = nn.Linear(out_channel_list[i] * block_type.expansion, generator.out_Nfeature)

	def _make_layer(self, generator,block, out_channels, blocks, stride=1):
		downsample = None
		if stride != 1 or self.in_channels != out_channels * block.expansion:
			downsample = nn.Sequential(
		        nn.Conv1d(self.in_channels, out_channels * block.expansion,
		                  kernel_size=1, stride=stride, bias=False),
		        nn.BatchNorm1d(out_channels * block.expansion),
		    )
		
		layers = []

		#--
		layers.append(block(generator,self.in_channels, out_channels, stride, downsample)) #call the block_typ
		#--

		self.in_channels = out_channels * block.expansion
		for _ in range(1, blocks):
			#--
			layers.append(block(generator,self.in_channels, out_channels)) #call the block_type
			#--
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.activation_func(x)
		x = self.maxpool(x)
	
		for layer in self.layers:	
			x = layer(x)
	#	x = self.layer1(x)
	#	x = self.layer2(x)
	#	x = self.layer3(x)
	#	x = self.layer4(x)
	#	x = self.layer5(x)
		
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x


# Example usage:
if __name__ == "__main__":

	g_p, args = f02_main()
	generator = Generator(g_p)

	model = ResNet1D(generator)
#	input_tensor = torch.randn(3, 1, 1000)  # Assuming input is (batch, channel, time)
	input_tensor = generator.generate_random()
	output = model(input_tensor)
	print(output.shape)
