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


from F01_loaddata import printmat,load_data,load_mnist,setup_batch_data
from F02_parameters import *
from F20_Discriminator import Discriminator, generate_random
from F10_Generator import *
from F40_SVM import SVM,LinearSVM
from F50_generator_loss import jensen_shannon_divergence


def shuffle_2d_tensor(input_tensor):
    """
    Shuffles the rows of a 2D Torch tensor.

    Args:
        input_tensor (torch.Tensor): The 2D input tensor to be shuffled.

    Returns:
        torch.Tensor: A new 2D tensor with the rows of input_tensor shuffled.
    """
    # Ensure the input is a 2D tensor
    assert input_tensor.dim() == 2, "Input tensor must be 2D."

    # Generate a random permutation of indices for the first dimension (M)
    perm_indices = torch.randperm(input_tensor.size(0))

    # Shuffle the tensor by reordering the rows using the permutation indices
    shuffled_tensor = input_tensor[perm_indices]

    return shuffled_tensor


#torch.autograd.set_detect_anomaly(True)
#cluster analysis perhaps

def GA_SVM(generator, SVM, dataset,inp_dim,subset_size=100):
	Epoch = generator.info.Epoch
	batch_size = generator.batch_size
	micro_iter = 1#generator.info.maxit

#	input_tensor = generate_random(inp_dim)

#	real_targets = torch.FloatTensor([1])
	target_zero = torch.FloatTensor([0])
	target_one = torch.FloatTensor([1])


	yreal = torch.ones(batch_size)
	yfake = torch.zeros(batch_size)
	ylabel = torch.cat((yreal,yfake))


	svm_loss=[]
	generator_loss=[]
	i=0

#	dataset_size = len(dataset)
	svm_optimizer = torch.optim.SGD([SVM.weight, SVM.bias], lr=0.001)

	threshold = torch.FloatTensor([2.0]) #the threshold to tell difference for cluster analysis


	coef = 2 #0.1
	#c_value = 0.05#0.01#5 #0.5 0.1
	clip_value=5000

	latent_dim = generator.info.latent_dim
	
#	print("input tensor use latent dim\n")

	gen_data = generator.generate_random(batch_size, inp_dim)
#	gen_data = generator.generate_random(batch_size, latent_dim)


	for epoch in range(Epoch):


		random.shuffle(dataset)
		batch_tensor_list = setup_batch_data(dataset, batch_size)
	#	print(len(batch_tensor_list)) #937
	#	print(subset_size)
		batch_tensor_list = random.sample(batch_tensor_list, subset_size)
		for idx,batch_tensor in enumerate(batch_tensor_list): 
			#train SVM
#			print(gen_data.shape)
#			print(batch_tensor.shape) #batch_tensor shape = [batch_size, inp_dim]
#			batch_tensor = shuffle_2d_tensor(batch_tensor)
			real_fake = torch.cat([batch_tensor,gen_data.detach()])
#			print(real_fake.shape)
#			y = ylabel[idx]
			svm_optimizer.zero_grad()
			#calc hinge loss: max(0, 1 - y*(wx + b))
			hinge_loss = torch.max(torch.tensor(0),1-ylabel*SVM(real_fake))
			#add normalization term: C * ||w||^2
			loss0 = hinge_loss+coef*torch.norm(SVM.weight)**2   #minimize ||w||
			#loss = torch.linalg.norm(loss0)
			loss = torch.sum(loss0)
			loss.backward()
			svm_optimizer.step()
	
		svm_loss.append(loss.item())
	

		#train generator
		generator.optimiser.zero_grad()
		#generate fake data
		gen_data = generator.generate_random(batch_size, inp_dim)
		gen_data = generator.forward(gen_data)

		#=============
		#attempt1 cluster
		g_loss1 = generator.loss1(gen_data, threshold)  #cluster
		#attempt2
		#g_loss2 = generator.loss2(gen_data, batch_tensor) # JS div
		##attempt3 KS div, code not working well
		##g_loss = generator.loss3(gen_data, batch_tensor) # 
		#attempt4 quantile distribution
		#g_loss4 = generator.loss4(gen_data, batch_tensor)
		#attempt5 fit vector
		#g_loss5 = generator.loss5(gen_data, batch_tensor)
		#attempt6 vector angle
		#g_loss6 = generator.loss6(gen_data, batch_tensor)
		#==========
		#g_loss = 0.6*g_loss1 + 0.1*g_loss2 + 0.1*g_loss4+ 0.1*g_loss5 + 0.1*g_loss6
		g_loss = g_loss1 
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
	                "[Epoch %d/%d] [SVM loss: %f] [G loss: %f]  "
	                 % (epoch, Epoch,  loss.item(), g_loss.item()))
	
	

#
#	print(svm_loss)
	SVM.plot_progress(svm_loss,"svm_loss")
	generator.plot_progress(generator_loss,"g_loss")

	return generator


def worker(flag=1):
	start_time = time.time()
	if flag==1:

		Energy_list, Geom_list, GE_list,tensor_geom_list = load_data(print_flag=0)
		g_p, d_p,args = f02_main()

		generator = Generator(g_p)
#		discriminator = Discriminator(d_p)
		inp_dim = generator.inp_dim #discriminator.inp_dim

#		svm = SVM()
		svm2 = LinearSVM(inp_dim)

		dataset_size=14000
		subset = random.sample(GE_list,dataset_size)
		trained_G = GA_SVM(generator,svm2,subset,inp_dim)

		pth_file = "generator.pth"
		print("pth file save to ", pth_file)
		torch.save(trained_G,pth_file)
		os.system("python3.6 A11_call_generator.py")
	
	if flag==2:
		print("load mnist")
		t1=time.time()
		mnist_data,mnist_datalist = load_mnist()	
		t2=time.time()
		print("load mnist time",t2-t1,"sec")
		g_p, d_p,args = f02_main()
		inp_dim = 28*28
		batch_size = g_p.batch_size
		g_p.inp_dim = inp_dim
#		g_p.G_batch_size = batch_size

		print(g_p)

		generator = Generator(g_p)

		svm2 = LinearSVM(inp_dim)

		subset_size=450 #(batches)  #12800#9600
#		subset = random.sample(mnist_datalist,dataset_size)

#		batch_tensor_list = setup_batch_data(subset, batch_size)

		print("start training")
		trained_G = GA_SVM(generator,svm2,mnist_datalist,inp_dim,subset_size)

		pth_file = "mnist.pth"
		print("pth file save to ", pth_file)
		torch.save(trained_G,pth_file)




		end_time = time.time()
		elapsed_time = end_time - start_time
		print(f"{elapsed_time:.6f} sec")
	
		os.system("python3.6 A12_call_mnist_generator.py")
	
	

if __name__=="__main__":
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
