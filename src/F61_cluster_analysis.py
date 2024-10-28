import numpy as np
import torch
import pandas
import matplotlib.pyplot as plt
import time
import multiprocessing  
from multiprocessing import Pool  
import copy
import random
from scipy.optimize import least_squares
from sklearn.cluster import KMeans

from F01_loaddata import printmat,load_data,load_mnist
from F02_parameters import *
from F20_Discriminator import Discriminator, generate_random
from F10_Generator import Generator
from F40_SVM import SVM,LinearSVM

#torch.autograd.set_detect_anomaly(True)
#cluster analysis perhaps

#def kmeans(X, n_clusters=1000, max_iters=100):
#    # 初始化质心，这里简单地取前n_clusters个样本作为初始质心
#    centroids = X[:n_clusters]
#    
#    for _ in range(max_iters):
#        # 计算每个样本到各个质心的距离
#        distances = torch.cdist(X, centroids)
#        
#        # 将每个样本分配给最近的质心
#        labels = torch.argmin(distances, dim=1)
#        
#        # 重新计算质心
#        new_centroids = torch.zeros_like(centroids)
#        for i in range(n_clusters):
#            new_centroids[i] = X[labels == i].mean(dim=0)
#            
#        # 如果质心没有变化，则结束迭代
#        if torch.allclose(new_centroids, centroids):
#            break
#            
#        centroids = new_centroids
#    
#    return labels, centroids

def Kmeans(dataset, Ncenter=1000,flag=2):
	if flag==1:
		combined_set = torch.stack(dataset).numpy()
	if flag==2:
		combined_set = torch.cat(dataset).numpy()	
	
	kmeans = KMeans(n_clusters=Ncenter, init='k-means++')
	kmeans.fit(combined_set)
	labels = kmeans.labels_
	centers = kmeans.cluster_centers_

	return labels, centers

def worker(flag=1):
	start_time = time.time()

	do_kmeans=True

	if flag==1:
		print("FAD data")
		Energy_list, Geom_list, GE_list,tensor_geom_list = load_data(print_flag=0)

		if do_kmeans:
			ncenter=2000
			labels,centers = Kmeans(GE_list,Ncenter=ncenter,flag=flag)
			np.savetxt(f"FAD_centers_{ncenter}.csv",centers)
		


	if flag==2:
		print("load mnist")
		mnist_data,mnist_datalist = load_mnist()	


		if do_kmeans:
			print("do kmeans")
			ncenter=10000
			labels,centers = Kmeans(mnist_datalist,Ncenter=ncenter,flag=flag)
			np.savetxt(f"mnist_centers_{ncenter}.csv",centers)
		else:
			centers = np.loadtxt("mnist_centers.csv")





	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"{elapsed_time:.6f} sec")


	

if __name__=="__main__":
	attempt=0
	if attempt==0:
		worker()
