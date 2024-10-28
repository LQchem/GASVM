import torch
from torch.autograd import Variable
import os
import numpy as np
import argparse
import torch.nn as nn
import copy
import random
import pandas as pd
import json

from F01_loaddata import *
def ConvertTensor(vec,grad_flag=True):
	return torch.tensor(vec,dtype=torch.float32,requires_grad=grad_flag)

def save_params(parser,args):
    """保存非默认值的参数到文件"""
    # 获取默认参数
    defaults = vars(parser.parse_args([]))
    params_to_save = {}

    # 遍历所有参数，只保存非默认值
    for key, value in vars(args).items():
        if value != defaults[key]:
            params_to_save[key] = value

    # 保存到文件
    with open('params.json', 'w') as f:
        json.dump(params_to_save, f)
    os.system("cp params.json backup")

class G_Parameter():
	def __init__(self):
		activation_funcs=[nn.ReLU(), #0    [0,inf]
			              nn.Tanh(), #1    [-1,1]
			              nn.Sigmoid(),#2  [0,1)
			              nn.Softmax(),#3  [0,1)
			              nn.ReLU6(),#4
			              nn.LeakyReLU(),#5 [-inf,inf]
						  nn.LeakyReLU(0.5) #6
	                      ]
		loss_funcs=[nn.MSELoss(),             nn.BCELoss(),           nn.L1Loss(),
			    nn.NLLLoss(),             nn.PoissonNLLLoss(),    nn.KLDivLoss(),
                            nn.BCEWithLogitsLoss(),   nn.MarginRankingLoss(),
                            nn.HingeEmbeddingLoss(),  nn.MultiLabelMarginLoss(),
                            nn.SmoothL1Loss(),        nn.SoftMarginLoss(),
                            nn.CosineEmbeddingLoss(), nn.MultiLabelSoftMarginLoss(),
                            nn.MultiMarginLoss(),     nn.TripletMarginLoss(),
                            nn.CTCLoss()]
	
		optimizers=[torch.optim.Adam,#0
                    torch.optim.SGD,#1
                    torch.optim.NAdam,#2
                    torch.optim.RAdam,#3
                    torch.optim.LBFGS,#4
		    torch.optim.RMSprop,#5
			torch.optim.Adadelta, #6
			torch.optim.Adagrad,#7
			torch.optim.AdamW,#8
			torch.optim.SparseAdam,#9
			torch.optim.Adamax,#10
			torch.optim.ASGD,#11
			torch.optim.Rprop#12
                      ]
		Normalization_method=[None,nn.BatchNorm1d,nn.LayerNorm]


		args  = ParseInp()
		self.args      = args 


		self.activation_func1 = activation_funcs[ self.args.G_activation_func1 ]
		self.activation_func2 = activation_funcs[ self.args.G_activation_func2 ]


		self.loss_func       = loss_funcs[ self.args.G_loss_func ]
		self.loss_model      = self.args.G_loss_model
		self.optimizer       = optimizers[ self.args.G_optimizer ]  #choose the optimizer
		self.target_dim      = self.args.G_target_dim 
		self.increment1      = self.args.G_increment1
		self.increment2      = self.args.G_increment2
		self.Nlayer          = self.args.G_Nlayer
		self.model_flag      = self.args.G_model_flag
		self.batch_size      = self.args.G_batch_size
		self.lr	             = self.args.G_lr
		self.Normalizer      = Normalization_method[self.args.G_norm_method]
		self.maxit           = self.args.G_maxit
		self.penalty_flag    = self.args.G_penalty_flag
		self.out_Nfeature    = self.args.G_out_Nfeature 
		self.NAtom           = self.args.M_NAtom
		self.loss_coef       = self.args.G_loss_coef

		#con1d or 2d	
		self.in_channel       = self.args.G_in_channel
		self.out_channel      = self.args.G_out_channel
		self.kernel_size      = self.args.G_kernel_size
		self.stride           = self.args.G_stride
		self.padding          = self.args.G_padding
		self.dilation         = self.args.G_dilation
		self.pool_kernel_size = self.args.G_pool_kernel_size
		self.pool_stride      = self.args.G_pool_stride 		
		#resnet1d
		self.block_type       = self.args.G_block_type #basic block or bottleneck for resnet1d
		self.num_block        = self.args.G_num_block
		self.f22_out_channel  = self.args.G_f22_out_channel
		self.f22_stride       = self.args.G_f22_stride		


		#2d
		self.expansion        = self.args.G_expansion 
		self.in_channels      = self.args.G_in_channels 
		self.out_channel1     = self.args.G_out_channel1 
		self.out_channel2     = self.args.G_out_channel2 





#		if self.Normalization == None:
#			pass
#		else: #the negative flag has normalization
#			self.model_flag  = self.model_flag+100

		#---------------------------------#
		#miscellaneous options
		self.thresh1         = self.args.M_thresh1
		self.thresh2         = self.args.M_thresh2
		self.avg_bond_length = self.args.M_avg_bond_length
		self.Epoch           = self.args.M_Epoch
		self.kmeans_file     = os.path.join("../csvfiles",self.args.M_kmeans_file)

#	#generator
#	def __str__(self):
#		print("Object of class G_Parameter:\n")
#		print("activation_function = ", self.activation_func)	
#		print("activation_function2 = ", self.activation_func2)	
#		print("loss_function       = ", self.loss_func)
#		print("optimizer           = ", self.optimizer)
#		print("learning rate       = ", self.lr,"\n")
#		print("Normalizer          = ", self.Normalization)
#		print("model flag          = ", self.model_flag)
#		print("increment           = ", self.increment)
#		print("increment2          = ", self.increment2)
#		print("Nlayer              = ", self.Nlayer,"\n")
#		print("target_dim          = ", self.target_dim,"\n")
#		print("G maxit             = ", self.maxit)
#		print("G penalty flag      = ", self.penalty_flag)
#		print("G inp_dim           = ", self.inp_dim)
#		print("Kmeans file         = ", self.kmeans_file)
#	#	print("ref CTable file     = ", self.ref_CTable_file,"\n")
#	#	print("NAtom               = ", self.NAtom)
#	#	print("bond thresh 4 CTable (Ang) = ", self.thresh)
#			
#
#		return "-----------------------------------------------------\n"
#
def ParseInp(json_flag=0):

#	 parser.add_argument('command',help="'train' or 'evaluate'")

    # Parse command line arguments
	parser = argparse.ArgumentParser(description='myGAN')

	parser.add_argument('--G_activation_func1', type=int,default=4) # and try 6
	parser.add_argument('--G_activation_func2', type=int, default=1) #1 tanh 2 sigmoid
	parser.add_argument('--G_loss_func', type=int,default=0)
	parser.add_argument('--G_loss_model',type=int,default=9)
	parser.add_argument('--G_optimizer', type=int, default=5)
	parser.add_argument('--G_model_flag', type=int,default=121) #101 use FC; 111 use conv1d; 121 conv1d w/ resnet;  #201 use resnet/conv2d for mnist; 251 use resnet for my data
	parser.add_argument('--G_target_dim', type=int, default=34) #here 
	parser.add_argument('--G_Nlayer', type=int,  default=3)
	parser.add_argument('--G_increment1', type=int, default=3)
	parser.add_argument('--G_increment2', type=int, default=0)
	#conv1d and conv2d
	parser.add_argument('--G_in_channel', type=int, default=1)
	parser.add_argument('--G_out_channel', type=int, default=3)
	parser.add_argument('--G_kernel_size', type=int,default=3)
	parser.add_argument('--G_stride', type=int, default=1)
	parser.add_argument('--G_padding', type=int, default=1)
	parser.add_argument('--G_dilation', type=int,default=1)
	parser.add_argument('--G_pool_kernel_size', type=int, default=1)
	parser.add_argument('--G_pool_stride', type=int,default=1)

	#resnet1d	
	parser.add_argument('--G_block_type', type=int,default=2)
	parser.add_argument('--G_num_block', type=int,default=2) #number of blocks in each layer in the resnet1d
	parser.add_argument('--G_f22_out_channel', type=int,default=3)
	parser.add_argument('--G_f22_stride', type=int,default=3)



	#conv2d
	parser.add_argument('--G_expansion', type=int, default=1)
	parser.add_argument('--G_in_channels', type=int, default=1)
	parser.add_argument('--G_out_channel1', type=int, default=1)
	parser.add_argument('--G_out_channel2', type=int, default=1)





	parser.add_argument('--G_loss_coef', type=float, default=0.5) #the weight betwen loss8 and loss9
	parser.add_argument('--G_batch_size', type=int, default=64)
	parser.add_argument('--G_lr', type=float, default=0.001)
	parser.add_argument('--G_norm_method',type=int,default=1) #0 not use normalization, 1 BatchNorm1d,2 LayerNorm
	parser.add_argument('--G_maxit',type=int,default=1) 
	parser.add_argument('--G_penalty_flag',type=int,default=50) #1 not use penalty 
	parser.add_argument('--G_out_Nfeature',type=int,default=34) 


	parser.add_argument('--M_thresh1', type=float, default=1.6)  # to detect if atoms are too close
	parser.add_argument('--M_thresh2', type=float, default=12.6) #to detect if atoms are too far
	parser.add_argument('--M_avg_bond_length',type=float,default=2.1)#estimated average bond length
	parser.add_argument('--M_Epoch', type=int,default=200)
	parser.add_argument('--M_kmeans_file', type=str,default="mnist_centers.csv")

	parser.add_argument('--M_NAtom',type=int,default=10)
	parser.add_argument('--M_Nconfig',type=int,default=14728)
	args= parser.parse_args()

	if json_flag==1:
		save_params(parser,args)
#	else:
#		print(json_flag,"test_")

	return args

class test(nn.Module):
	def __init__(self,args):
		self.load_para(args)

	def load_para(self,args):
		for attr_name, attr_value in vars(args).items():
			setattr(self,attr_name,attr_value)

def f02_main(json_flag=1):
	args = ParseInp(json_flag)
	g_p = G_Parameter()
#	print(g_p)	
#	print(d_p)
	return g_p, args
if __name__=="__main__":
	g_p, args = f02_main()
	t=test(args)
	print(t.__dict__)
