import numpy as np
import torch
import pandas
import torchvision
import torch.nn as nn


import matplotlib.pyplot as plt
from F01_loaddata import *
from F02_parameters import *
from F11_Generator_reformat import *
def print_generator(self):
		print("Object of class G_Parameter:\n")
		print("activation_function1 = ", self.activation_func1)	
		print("activation_function2 = ", self.activation_func2)	
		print("batch_size           = ", self.batch_size)
		print("block type           = ", self.block_type)
		print("Epoch                = ", self.Epoch,"\n")
		print("increment1           = ", self.increment1)
		print("increment2           = ", self.increment2)
		print("Kmeans file          = ", self.kmeans_file)
		print("loss_function        = ", self.loss_func)
		print("learning rate        = ", self.lr,"\n")
		print("model flag           = ", self.model_flag)
		print("num_blcok            = ", self.num_block)
		print("Normalizer           = ", self.Normalizer)
		print("Nlayer               = ", self.Nlayer)
		print("optimizer            = ", self.optimizer)
		print("out_channel (f22)    = ", self.f22_out_channel)
		print("stride (f22)         = ", self.f22_stride)
		print("target_dim           = ", self.target_dim)


		print("G out_N_feature      = ", self.out_Nfeature)
		print("G maxit              = ", self.maxit)
		print("G penalty flag       = ", self.penalty_flag)
	#	print("ref CTable file      = ", self.ref_CTable_file,"\n")
	#	print("NAtom                = ", self.NAtom)
	#	print("bond thresh 4 CTable (Ang) = ", self.thresh)
			

		print("-----------------------------------------------------\n")


#read in [nbatch, Nfeature]
#write out [nbatch,Nfeature]
def Linear_Model1(self):
	Nlayer     = self.Nlayer
	target_dim = self.target_dim
	activation_func1 = self.activation_func1
	activation_func2 = self.activation_func2
	Normalizer  = self.Normalizer
	increment1  = self.increment1
	out_feature = self.out_Nfeature 

	#lqnote transform/scale generated data
	module_list=[]
	for i in range(Nlayer):
		module_list.append(nn.Linear(target_dim,target_dim+increment1))
		module_list.append(activation_func1)
		module_list.append(Normalizer(target_dim+increment1))
		target_dim+=increment1
	
	for i in range(Nlayer):
		module_list.append(nn.Linear(target_dim,target_dim-increment1))
		module_list.append(activation_func1)
		module_list.append(Normalizer(target_dim-increment1))
		target_dim-=increment1
	
	module_list.append(nn.Linear(target_dim,out_feature))
#	module_list.append(activation_func1)
	model = nn.Sequential(*module_list)

	return model  

def Linear_Model2(self):
	target_dim = self.target_dim
	out_feature = self.out_Nfeature 

	module_list=[]
	module_list.append(nn.Linear(target_dim,out_feature))
	model = nn.Sequential(*module_list)

	return model


def Linear_Model3(self):
	Nlayer     = self.Nlayer
	target_dim = self.target_dim
	activation_func1 = self.activation_func1
	activation_func2 = self.activation_func2
	Normalizer  = self.Normalizer
	increment1  = self.increment1
	out_feature = self.out_Nfeature 

	#lqnote transform/scale generated data
	module_list=[]
	for i in range(Nlayer):
		module_list.append(nn.Linear(target_dim,target_dim+increment1))
		module_list.append(activation_func1)
#		module_list.append(Normalizer(target_dim+increment1))
		target_dim+=increment1
	
	for i in range(Nlayer):
		module_list.append(nn.Linear(target_dim,target_dim-increment1))
		module_list.append(activation_func1)
#		module_list.append(Normalizer(target_dim-increment1))
		target_dim-=increment1
	
	module_list.append(nn.Linear(target_dim,out_feature))
	module_list.append(activation_func1)
	model = nn.Sequential(*module_list)

	return model  


#on input:
#L, out_feature
#K, kernel_size
#S, stride
#P, padding
#D, dilation
#Kpool, pool_kernel_size
#Spool, pool_Stride
#Ppool, pool_padding
def calc_convo1D_Nfeature(self,L,K, S, P, D, Kpool=1, Spool=1, Ppool=0):
	#calc conv1d Nfeature without pool
	Lout = (L+2*P-K)/S + 1

	#if include pool
	Lout = (Lout+2*Ppool-Kpool)/Spool + 1
	return int(Lout)


#read in [nbatch, nchannel, Nfeature]
#writeout [[nbatch, nchannel, Nfeature]
def Convo1d_Model1(self): #model 111
	Nlayer     = self.Nlayer
	target_dim = self.target_dim
	activation_func1 = self.activation_func1
	activation_func2 = self.activation_func2
	Normalizer  = self.Normalizer
	out_feature = self.out_Nfeature	
#	increment1 = self.increment1
	batch_size = self.batch_size

	in_channel  = self.in_channel
	out_channel = self.out_channel
	kernel_size = self.kernel_size
	stride      = self.stride
	padding     = self.padding
	dilation    = self.dilation

	pool_kernel_size = self.pool_kernel_size
	pool_stride      = self.pool_stride
	module_list=[]


	for _ in range(Nlayer):
		module_list.append(nn.Conv1d(in_channels = in_channel,out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation))
		module_list.append(activation_func1)
		module_list.append(Normalizer(out_channel))	
		module_list.append(nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride))
		in_channel = out_channel

#	module_list.append(Normalizer(self.target_dim))
	model = nn.Sequential(*module_list)
	return model
		


#-----reproduce resnet-------------------
def block(self,in_channels, out_channels,stride=1):
	module_list = []

	kernel_size = self.kernel_size
	padding     = self.padding
	expansion   = self.expansion

	#content of the block
	conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias=False)
	bn1   = nn.BatchNorm2d(out_channels)
	conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias=False)
	bn2   = nn.BatchNorm2d(out_channels)

	#setup the shortcut
	shortcut = nn.Sequential()
	if stride != 1 or in_channels != expansion * out_channels:
		shortcut = nn.Sequential(
                nn.Conv2d(in_channels, expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expansion * out_channels)
            )

	module_list.append(conv1)
	module_list.append(bn1)
	module_list.append(conv2)
	module_list.append(bn2)
	module_list.append(shortcut)
	block = nn.Sequential(*module_list)
#	block.append(shortcut)
	return block


def make_layer(self, out_channels, num_blocks, stride):
	strides = [stride] + [1] * (num_blocks - 1) #strids is a list of 2 members:[stride,  1*(num_blocks-1)]
	layers = []
	in_channels = self.in_channels
	for stride in strides:
		layers.append(block(generator, in_channels,out_channels, stride))
		in_channels = out_channels * self.expansion
	return layers

def calculate_final_size(input_size,stride_list):
	'''
	计算经过多次下采样后的特征图的最终尺寸。
	'''
	h, w = input_size
	for stride in stride_list:
		h = h // stride
		w = w // stride
	return h,w

def myResnet(self):
	in_channels  = self.in_channels
	out_channel1 = self.out_channel1
	out_channel2 = self.out_channel2
	Nlayer       = self.Nlayer

	kernel_size = self.kernel_size
	stride      = self.stride
	padding     = self.padding
	expansion   = self.expansion

	conv1 = nn.Conv2d(in_channels, out_channel1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
	bn1   = nn.BatchNorm2d(out_channel1)

	layers = []
	
	#TODO here is simplified
	# 定义 ResNet 的各个阶段
	num_block = 3 #dfine how many blocks are included 
	stride_list=[1,2]
	layer1 = make_layer(self, out_channel1,num_block, stride_list[0])
	layer2 = make_layer(self, out_channel2,num_block, stride_list[1])
	
	layers.extend(layer1)
	layers.extend(layer2)



	# 动态计算特征图的最终尺寸
	fig_size = [self.target_dim,self.target_dim]
	final_size = calculate_final_size(fig_size,stride_list)

	# 设置平均池化的大小
	layers.append(nn.AvgPool2d(final_size))
	layers.append(nn.Flatten())
	layers.append(nn.Linear(out_channel2 * self.expansion, self.out_Nfeature))		

	model = nn.Sequential(*layers) 
	return model
#----end of my resnet-----------------------



if __name__=="__main__":
	shift_value=379
#	energy_tensor, shiftE_tensor,dipole_tensor, geom_list,Nconfig, atoms = load_data(shift_value)	
#	NAtom = 10
	
#	GEDlist = setup_tensor_list(shiftE_tensor, dipole_tensor,geom_list,NAtom)

	g_p, args = f02_main()
	
#	print(g_p)

	generator = Generator(g_p)

	module_list = block(generator, 1,6)
	print("----")
#	print(module_list)

#	print("123123")
#	make_layer(generator, 4, 2,1)

	model = myResnet(generator)	
	print(model)
##	print(generator)


