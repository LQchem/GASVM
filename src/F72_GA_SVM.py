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


from F01_loaddata import * 
from F02_parameters import *
from F03_feature import *
from F11_Generator_reformat import *
from F12_G_models import *
from F21_resnet import *
from F40_SVM import LinearSVM
from F50_generator_loss import jensen_shannon_divergence
#from F65_PES_NN import do_normalization


#torch.autograd.set_detect_anomaly(True)
def count_paremeters(model):
	num = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("total number of model parameters ", num) 


def GA_SVM(generator, SVM, dataloader):
	Epoch = generator.Epoch
	batch_size = generator.batch_size
	micro_iter = 1#generator.info.maxit

	target_zero = torch.FloatTensor([0])
	target_one = torch.FloatTensor([1])


	yreal = torch.ones(batch_size)
	yfake = -yreal #torch.zeros(batch_size)
	ylabel = torch.cat((yreal,yfake))


	svm_loss=[]
	generator_loss=[]
	i=0

	svm_optimizer = torch.optim.SGD([SVM.weight, SVM.bias], lr=0.001)


	coef = 0.1
	#c_value = 0.05#0.01#5 #0.5 0.1
	clip_value=5000

	target_dim = generator.target_dim
	
#	print("input tensor use latent dim\n")

#	gen_data = generator.generate_random(batch_size, latent_dim)
#	gen_data = generator.forward(gen_data)

	
	loss_coef = 0.45 #0.8 #0.7 0.9#generator.loss_coef



	for epoch in range(Epoch):
		for idx,batch_tensor in enumerate(dataloader):  #batch_tensor is of 3N+1+3 if linear NN is used


	
			#train generator
			generator.optimiser.zero_grad()
			#generate fake data
			gen_data = generator.generate_random(batch_size, target_dim)  #gen_data is of shape [batch_size, 1, target_dim]

			#---
			#lq patch
			#gen_data[:,:,-4] += -3 
			#---lq patch

			gen_data = generator.forward(gen_data)

	#		normalized_batch_tensor, norms1 = do_normalization(batch_tensor)
	#		normalized_gen_data,nomrs2 = do_normalization(gen_data)

			#=============
			g_loss_cov_mat     = generator.loss9(generator,gen_data,batch_tensor)
			g_loss_center_diff = generator.loss10(generator,gen_data, batch_tensor) 
			g_loss = loss_coef * g_loss_cov_mat + (1-loss_coef) * g_loss_center_diff 
			#==========

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
	
		#train SVM

			real_fake = torch.cat([batch_tensor,gen_data.detach()]) #real_fake of shape (batch_size*2, out_feature)
			svm_optimizer.zero_grad()
			#calc hinge loss: max(0, 1 - y*(wx + b))
			hinge_loss = torch.max(torch.tensor(0),1-ylabel*SVM(real_fake)) #SVM(real_fake) of shape batch_size*2, same shape after multiplying ylabel
			#add normalization term: C * ||w||^2
			loss0 = hinge_loss+coef*torch.norm(SVM.weight)**2   #minimize ||w||
			#loss = torch.linalg.norm(loss0)
			loss = torch.sum(loss0)
			loss.backward()
			svm_optimizer.step()
	
			svm_loss.append(loss.item())
	


		print(
	                "[Epoch %d/%d] [SVM loss: %f] [G loss: %f]  "
	                 % (epoch, Epoch,  loss.item(), g_loss.item()))
	
	

#
#	print(svm_loss)
	SVM.plot_progress(svm_loss,"svm_loss")
	generator.plot_progress(generator_loss,"g_loss")

	return generator


def worker(flag=1):
	flag=1
	start_time = time.time()
	if flag==1:
		'''
		This attempt use dataloader,designed either for LNN or CNN
		'''
		print("load my data")
		shift_value=373

		feature = Feature(shift_value)

		feature_type=2
		if feature_type==0: #not in use
			#---------------------
			#   use cartesian coord and energy and dipole as features
			#	energy_tensor, shiftE_tensor,dipole_tensor, geom_list,Nconfig, atoms = load_data(shift_value)
			#	#setup the tensor list for LNN, whose element's size is of (NAtom*3+4) 
			#   not use the Feature class
				GEDlist2,scaled_GEDlist2 = setup_tensor_list2(shiftE_tensor, dipole_tensor,geom_list)
				my_datalist = GEDlist2
				out_feature = GEDlist2[0].shape[0]
			#-----------------------

		if feature_type==1:
			#use bond map upper triA for monomers and inter-mol dist as features
			#my_datalist = feature.feature_list   		
			print("use bond map (triA and inter-mol dist)")
			feature.select_feature_type(f_type=feature_type)
			my_datalist = feature.features
			out_feature = len(my_datalist[0])	#len = 10+10+25
			print("out feature",out_feature)
			cmd="python3.6 A14_call_generator_bondmap.py"

		if feature_type==2:
			#use Feature class and use GED as features; energy  is shifted
			print("use GED as features")
			feature.select_feature_type(f_type=feature_type)
			my_datalist = feature.features 
			out_feature = len(my_datalist[0])   #len = 3N+1+3
			print("out feature",out_feature)
			cmd="python3.6 A13_call_generator.py"


		if feature_type==3:
			#use 2d geom as features
			print("use 2d geometry mat as features")
			feature.select_feature_type(f_type=feature_type)	 #energy is shifted
			my_datalist = feature.features  #members in the featuers is a tensor of shape (3,NAtom,3), first channel G, 2nd channel E, 3rd channel D
	 
			out_feature = len(my_datalist[0])   #my_datalist[0] is of shape Natom*3, the out_feature is not use in resnet 
			print("out feature",out_feature)
			cmd="python3.6 A15_call_generator_resnet.py"


		g_p, args = f02_main(json_flag=1)
#		g_p.out_Nfeature = out_feature
		batch_size = g_p.batch_size


		#if do the normalization to the input data
		#my_datalist = do_normalization(my_datalist)


		dataloader_l = setup_dataloader(my_datalist,batch_size) #for linear NN
		
		generator = Generator(g_p)
		SVM       = LinearSVM(out_feature)

		count_paremeters(generator)


		trained_G = GA_SVM(generator,SVM, dataloader_l)

		pth_file = "generator.pth"
		print("pth file save to ", pth_file)
		torch.save(trained_G,pth_file)

		end_time = time.time()
		elapsed_time = end_time - start_time
		print(f"{elapsed_time:.6f} sec")

		print(cmd)
		os.system(cmd)


	

if __name__=="__main__":

	#python3.6 F72_GA_SVM.py  --G_Nlayer=3 --G_increment1=3 --G_num_block=2 --G_f22_out_channel=3 --G_activation_func1=4

	attempt=0
	if attempt==0:
		worker()
#	if attempt==1:
#		processes = []  
#		for _ in range(4):  # 创建2个进程  
#			p = multiprocessing.Process(target=worker)  
#			processes.append(p)  
#			p.start()  
#		
#		for p in processes:  
#			p.join()
#
#	if attempt==2:
#		Energy_list, Geom_list, GE_list = load_data(print_flag=0)
#		dataset_size=100
#		subset = random.sample(GE_list,dataset_size)
#	
#		with Pool() as pool: 
#			pool.map(worker,subset)
#
