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
from scipy.optimize import least_squares

from F01_loaddata import printmat,load_data,load_mnist
from F02_parameters import *
from F20_Discriminator import Discriminator, generate_random
from F10_Generator import Generator
from F40_SVM import SVM,LinearSVM

#torch.autograd.set_detect_anomaly(True)
#cluster analysis perhaps

def LinearFit(dataset):
	pass

#find the data closet to the target.
def find_closest_points_heap(lst, x, k=10):
	# 初始化一个最大堆，差值取负数以便得到最小值的效果
	max_heap = []
	for point in lst:
		diff = torch.linalg.norm(point - x)
		# 如果堆未满或新点更接近，则添加到堆中并保持堆的大小为k
		if len(max_heap) < k:
			heapq.heappush(max_heap, (-diff, point))
		else:
			heapq.heappushpop(max_heap, (-diff, point))
	
	# 从堆中提取点并返回，注意要取正值
	res = [point for (_, point) in sorted(max_heap)]	
	return res


def QuadraticFit(neighbors,ref_point):
	
	

	return res


def worker(flag=2):
	start_time = time.time()
	if flag==2:
		print("load mnist")
		mnist_data,mnist_datalist = load_mnist()	
		g_p, d_p,args = f02_main()
		inp_dim = 28*28
		g_p.inp_dim = inp_dim

		print(g_p)

		input_tensor = generate_random(inp_dim)


		generator = Generator(g_p)
		discriminator = Discriminator(d_p)

		gen_datasize=100
		Nneighbors =  10

		gen_list = [] 
		for _ in range(gen_datasize):
			fake_data = generator.forward(input_tensor).detach()
			gen_list.append(fake_data)
			neighbors = find_closest_points_heap(mnist_datalist, fake_data, k=Nneighbors)


		#svm2 = LinearSVM(inp_dim)

		#dataset_size=60000
		#subset = random.sample(mnist_datalist,dataset_size)
		#print("start training")
		#trained_G = GA_SVM(generator,svm2,subset,inp_dim)

		#pth_file = "mnist.pth"
		#print("pth file save to ", pth_file)
		#torch.save(trained_G,pth_file)




	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"{elapsed_time:.6f} sec")


	

if __name__=="__main__":
	attempt=0
	if attempt==0:
		worker()
