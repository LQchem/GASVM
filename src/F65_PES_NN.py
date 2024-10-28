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


def do_normalization(datalist):
	print("scaling the input data and normalize the input")
	#---if datalist is of type list---
	#max_val = torch.max(torch.stack([torch.max(tensor) for tensor in datalist]))
	#min_val = torch.min(torch.stack([torch.max(tensor) for tensor in datalist]))

	#normalized_tensor = [(tensor - min_val) / (max_val - min_val) for tensor in datalist]
	#---------------

	#---if datalist is of tensor type---
	norms = torch.norm(datalist,p=2,dim=1,keepdim=True)
	norms[norms==0] = 1 #to avoid dividing 0
	normalized_tensor = datalist / norms


	#to do the back normalization
	#orinigal_tensor = normalized_tensor * norms
	return normalized_tensor, norms



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

			e_loss = nn.MSELoss()(out[:,0],bigY[:,0])  #only compare energy
			d_loss = nn.MSELoss()(out[:,1:],bigY[:,1:])#only compare dipole
			g_loss = nn.MSELoss()(out,bigY) 
			
			
			#g_loss.backward()
			e_loss.backward()
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
	                "[Epoch %d/%d] [G loss: %f] [E loss: %f] [D loss: %f]  "
	                 % (epoch, Epoch,  g_loss.item(), e_loss.item(), d_loss.item()))
	
	

#	generator.plot_progress(generator_loss,"g_loss")

	return generator


def PES_test(test_list, NN_pes,shift=379):
#	print("run pes test")
#	NN_pes = torch.load(pth_file)

	E_error_list =[]
	dim = test_list.shape[0]
	for i in range(dim):
		data      = test_list[i,:]
		flat_geom = data[0:-4]
		energy    = data[-4]
		dipole    = data[-3:]

		tmp    = flat_geom.unsqueeze(0)
#		bigX   = torch.vstack([tmp, tmp])
		bigX   = tmp
		pred_out = NN_pes.forward(bigX)
		pred_E   = pred_out[0,0]-shift
		pred_D   = pred_out[0,1:]

		diffE = energy - pred_E
		E_error_list.append(diffE.detach().numpy())

	#calc rmse
	a = np.array(E_error_list)
	rmse = np.sqrt(np.mean((a-np.mean(a))**2))
	print("rmse between fitted Energy and real enerey", rmse)

	return rmse

def PES_test2(test_list, PES_NN, shift=379):
	test_tensor = ConvertTensor(test_list)
	test_geom   = test_tensor[:,0:-4] #real value
	test_E      = test_tensor[:,-4]   #real value
	test_D      = test_tensor[:,-3:]  #real value
 
	out = PES_NN.forward(test_geom)
	e_loss = nn.MSELoss()(out[:,0],test_E)  #only compare energy
	d_loss = nn.MSELoss()(out[:,1:],test_D)#only compare dipole

	print("MSE loss for Energy between real value and PES fitted value", e_loss.item())
	print("MSE loss for dipole between real value and PES fitted value", d_loss.item())

	return e_loss.item(), d_loss.item()


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
			my_datalist = feature.features  #if feature_type==2, my_datalist if of tensor type 
			out_feature = len(my_datalist[0])   #len = 3N+1+3
			print("out feature",out_feature)


		#if do the normalization to the input data
		my_datalist,norms = do_normalization(my_datalist) #thus both train and test sets are normalized


		g_p, args = f02_main(json_flag=0)

		g_p.model_flag = 103
		g_p.out_Nfeature = 4
		g_p.target_dim = 30

		batch_size = g_p.batch_size
		generator = Generator(g_p)
	

		truncate=1
		if truncate: #split the whole set into train and test subset
			train_ratio = 0.75
			#~~~
			subset_size = int( len(my_datalist)*train_ratio )
			idx = random.sample(range(len(my_datalist)),subset_size)
			remain_idx = set(list(range(len(my_datalist)))) - set(idx)
			remain_idx = list(remain_idx)
			subset = my_datalist[idx]
			remain = my_datalist[remain_idx]
			#~~~
			epoch = 20
	
			#lq
			pth_file = "PES_NN.pth"
			trained_G = torch.load(pth_file)
			PES_test2(remain,trained_G)

				
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
				eloss,dloss = PES_test2(remain,trained_G) #remain is of list type
				rmse_list.append(eloss)
	
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

#		#compare PES_NN results against GA_SVM resutls
#		cmd = "python3.6 A13_call_generator.py"
#		print(cmd)
#		os.system(cmd) 

	

if __name__=="__main__":
	'''
	train the xyz with E and D, with plain PES NN, to avoid calculation of generated points
	python3.6 F65_PES_NN.py --G_activation_func1=6 --G_increment1=0 --G_Nlayer=3
	'''


	attempt=0
	if attempt==0:
		worker()
