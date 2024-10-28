import numpy as np
import torch
import pandas
import matplotlib.pyplot as plt
import time
import multiprocessing  
from multiprocessing import Pool  
import copy
import random
import heapq
import os

from F01_loaddata import * 
from F02_parameters import *
from F03_feature import *
from F11_Generator_reformat import *
from F12_G_models import *
from F21_resnet import *
from F40_SVM import LinearSVM
from F50_generator_loss import jensen_shannon_divergence
def plot(data):
	dim = len(data)
	x= np.arange(dim)
	fig = plt.figure()
	plt.plot(x,data)
	plt.savefig("rmse.png")
	plt.show()
#	plt.close()
#torch.autograd.set_detect_anomaly(True)
def count_paremeters(model):
	num = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("total number of model parameters ", num) 


def PES_train(generator,  dataloader):
	Epoch = generator.Epoch
	batch_size = generator.batch_size
	micro_iter = 1#generator.info.maxit

	generator_loss=[]
	i=0


	coef = 0.1
	#c_value = 0.05#0.01#5 #0.5 0.1
	clip_value=5000

	target_dim = generator.target_dim
	
	
	loss_coef = generator.loss_coef
	for epoch in range(Epoch):
		for idx,batch_tensor in enumerate(dataloader):  
			bigY = batch_tensor[:,-4:]
			bigX = batch_tensor[:,0:-4]
			#print(bigY.shape,bigX.shape)		
	
			#train generator
			generator.optimiser.zero_grad()

			out = generator.forward(bigX)
			g_loss = nn.MSELoss()(out,bigY) 

			g_loss.backward()
			generator.optimiser.step()
	
			generator_loss.append(g_loss.item())
	
	
			#---------------------
			#WGAN
			#c_value = 0.1 - epoch/clip_value
			#if c_value < 0.01:
			#	c_value = 0.01
	
	
			#for p in generator.parameters():
			#	p.data.clamp_(-c_value,c_value)
			#-----------------------------
	

		print(
	                "[Epoch %d/%d] [G loss: %f]  "
	                 % (epoch, Epoch,  g_loss.item()))
	
	

#	generator.plot_progress(generator_loss,"g_loss")

	return generator


def PES_test(test_list, generator_pes,shift=379):
#	print("run pes test")
#	generator_pes = torch.load(pth_file)

	E_error_list =[]
	dim = test_list.shape[0]
	for i in range(dim):
		data      = test_list[i,:]
		flat_geom = data[0:-4]
		energy    = data[-4]
		dipole    = data[-3:]

		tmp    = flat_geom.unsqueeze(0)
		bigX   = torch.vstack([tmp, tmp])
		pred_out = generator_pes.forward(bigX)
		pred_E   = pred_out[0,0]-shift
		pred_D   = pred_out[0,1:]

		diffE = energy - pred_E
		E_error_list.append(diffE.detach().numpy())

	#calc rmse
	a = np.array(E_error_list)
	rmse = np.sqrt(np.mean((a-np.mean(a))**2))
	print("rmse between fitted Energy and real enerey", rmse)

	return rmse

def worker(flag=1):
	flag=1
	start_time = time.time()
	if flag==1:
		'''
		This attempt use dataloader,designed either for LNN or CNN
		'''
		print("load my data")
		shift_value=379

		feature = Feature(shift_value)

		feature_type=2

		if feature_type==2:
			#use Feature class and use GED as features; energy  is shifted
			print("use GED as features")
			feature.select_feature_type(f_type=feature_type)
			my_datalist = feature.features 
			out_feature = len(my_datalist[0])   #len = 3N+1+3
			print("out feature",out_feature)



		g_p, args = f02_main(json_flag=0)

		g_p.model_flag = 101
		g_p.out_Nfeature = 1
		g_p.target_dim = 30

		batch_size = g_p.batch_size
		generator = Generator(g_p)

		truncate=1
		if truncate: #split the whole set into train and test subset
			train_ratio = 0.7
			#~~~
			subset_size = int( len(my_datalist)*train_ratio )
			idx = random.sample(range(len(my_datalist)),subset_size)
			remain_idx = set(list(range(len(my_datalist)))) - set(idx)
			remain_idx = list(remain_idx)
			subset = my_datalist[idx]
			remain = my_datalist[remain_idx]
			#~~~
			epoch = 10
				
			#train the PES NN
			rmse_list = []
			for i in range(epoch):		
				print(i)
				#~~~
				subset_size = int( len(my_datalist)*train_ratio )
				idx = random.sample(range(len(my_datalist)),subset_size)
				remain_idx = set(list(range(len(my_datalist)))) - set(idx)
				remain_idx = list(remain_idx)
				subset = my_datalist[idx]
				remain = my_datalist[remain_idx]
				#~~~
	
				dataloader_l = setup_dataloader(my_datalist,batch_size)
				trained_G = PES_train(generator,dataloader_l)
				rmse = PES_test(remain,trained_G)
				rmse_list.append(rmse)
	
			plot(rmse_list)

		else:
			dataloader_l = setup_dataloader(my_datalist,batch_size)
			trained_G = PES_train(generator,dataloader_l)
				



		pth_file = "PES_NN.pth"
		print("pth file save to ", pth_file)
		torch.save(trained_G,pth_file)

		end_time = time.time()
		elapsed_time = end_time - start_time
		print(f"{elapsed_time:.6f} sec")

		#compare PES_NN results against GA_SVM resutls
		cmd = "python3.6 A17_call_generator_fromA13.py"
		print(cmd)
		os.system(cmd) 

	

if __name__=="__main__":
	'''
	copy from F65
	train the xyz with only E  with plain PES NN, to avoid calculation of generated points
	'''


	attempt=0
	if attempt==0:
		worker()
