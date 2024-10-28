import numpy as np
import torch
import pandas
import matplotlib.pyplot as plt
import time
import multiprocessing  
from multiprocessing import Pool  


from F01_loaddata import printmat,load_data
from F02_parameters import *
from F20_Discriminator import Discriminator, generate_random
from F10_Generator import Generator
import random

def GAN_train(generator, discriminator,dataset):

	Epoch = generator.info.Epoch
	micro_iter = generator.info.maxit

	inp_dim = discriminator.inp_dim
	input_tensor = generate_random(inp_dim)

	real_targets = torch.FloatTensor([1])
	fake_targets = torch.FloatTensor([0])


	discriminator_loss=[]
	generator_loss=[]
	i=0
	punish=0
	for epoch in range(Epoch):
		random.shuffle(dataset)
		for data in dataset:

			#train generator
			for m in range(micro_iter):
				gen_tensor = generator.forward(input_tensor)	
				d_output = discriminator.forward(gen_tensor)
				g_loss = generator.info.loss_func(d_output,real_targets)
				generator.optimiser.zero_grad()
				g_loss.backward()
				generator.optimiser.step()

			#train the discriminator
			discriminator.optimiser.zero_grad()
			real_data = ConvertTensor(data,False)
			d_res = discriminator(real_data)
			real_loss = discriminator.info.loss_func(d_res,real_targets)
		
			d_res = discriminator(gen_tensor.detach())
			fake_loss = discriminator.info.loss_func(d_res,fake_targets)

			d_loss = (real_loss + fake_loss) *0.5
			d_loss.backward()
			discriminator.optimiser.step()		
	

		discriminator_loss.append(d_loss.item())
		generator_loss.append(g_loss.item())
		i+=1
		print(
                 "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]  [punish: %f]"
                 % (epoch, Epoch, i,len(dataset), d_loss.item(), g_loss.item(), punish))



	discriminator.plot_progress(discriminator_loss,"d_loss")
	generator.plot_progress(generator_loss,"g_loss")

	return generator

def worker():
	start_time = time.time()
	Energy_list, Geom_list, GE_list = load_data(print_flag=0)
	g_p, d_p,args = f02_main()

	discriminator = Discriminator(d_p)
	generator = Generator(g_p)

	dataset_size=20
	subset = random.sample(GE_list,dataset_size)
	
	trained_G = GAN_train(generator,discriminator,subset)
	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"{elapsed_time:.6f} sec")

	pth_file = "generator.pth"
	print("pth file save to ", pth_file)
	torch.save(trained_G,pth_file)


	

if __name__=="__main__":
	attempt=0
	if attempt==0:
		worker()
	if attempt==1:
		processes = []  
		for _ in range(4):  # 创建2个进程  
			p = multiprocessing.Process(target=worker)  
			processes.append(p)  
			p.start()  
		
		for p in processes:  
			p.join()

	if attempt==2:
		Energy_list, Geom_list, GE_list = load_data(print_flag=0)
		dataset_size=100
		subset = random.sample(GE_list,dataset_size)
	
		with Pool() as pool: 
			pool.map(worker,subset)

