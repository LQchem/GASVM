import numpy as np
import torch
import pandas
import torchvision
import matplotlib.pyplot as plt
from F01_loaddata import *
from F02_parameters import *
from F50_generator_loss import * 
def find_closet_center(gen_point, centers):
	Ncenter, Nfeature = centers.shape

	dist_2_center = 1000 #an arbitrily large value

	for j in range(Ncenter):
		center = centers[j,:]
		dist = torch.linalg.norm( center - gen_point)
		if dist < dist_2_center :
			dist_2_center = dist
	
	return dist_2_center

def cluster_analysis(gen_data,kmeans_file):

	centers = np.loadtxt(kmeans_file)
	centers = ConvertTensor(centers, False)

	accum_dist = 0
	for i in range(gen_data.shape[0]):
		accum_dist = find_closet_center(gen_data[i,:],centers)


	accum_dist.requires_grad_()
	return 	accum_dist



def loss1(self,gen_data, threshold=torch.FloatTensor([2.0]) ):
	accum_dist = cluster_analysis(gen_data,self.kmeans_file)
	g_loss = self.loss_func(accum_dist, threshold)	
	return g_loss

#JS divergence
def loss2(self,gen_data, ref_tensor):
	js_div = jensen_shannon_divergence(gen_data, ref_tensor)
	g_loss = self.loss_func(js_div, self.target_zero)
	return g_loss

#KS divergence #code not working, and may not be useful
def loss3(self, gen_data, ref_tensor):
	p_value = Kolmogorov_Smirnov_divergence(gen_data, ref_tensor)
	g_loss = self.loss_func(p_value, self.target_one)
	return g_loss

#quantile	
def loss4(self,gen_data, ref_tensor):
	q1 = torch.quantile(gen_data, 0.25,dim=0)
	q2 = torch.quantile(gen_data, 0.50,dim=0)
	q3 = torch.quantile(gen_data, 0.75,dim=0)

	p1 = torch.quantile(ref_tensor, 0.25,dim=0)
	p2 = torch.quantile(ref_tensor, 0.50,dim=0)
	p3 = torch.quantile(ref_tensor, 0.75,dim=0)

	l1 = self.loss_func(p1,q1)
	l2 = self.loss_func(p2,q2)
	l3 = self.loss_func(p3,q3)

	return l1+l2+l3


#overlap between two fit vector
def loss5(self,gen_data, ref_tensor,maxit=100):
	self.linear_fit1 = LinearFit(self.latent_dim) 
	self.linear_fit2 = LinearFit(self.latent_dim) 


	self.linear_fit1.load_data(ref_tensor.detach())
	self.linear_fit2.load_data(gen_data.detach())

	# Fit the models but don't compute gradients in this custom loss function
#	with torch.no_grad():
	w1, b1 = self.linear_fit1.fit(num_epochs = maxit) 
	w2, b2 = self.linear_fit2.fit(num_epochs = maxit)

	g_loss = self.loss_func(w1, w2) + self.info.loss_func(b1,b2)
	return g_loss

#vector angle
def loss6(self, gen_data, ref_tensor):
	mean1 = torch.mean(gen_data, dim=0)
	mean2 = torch.mean(ref_tensor, dim=0)

#	print(mean1.shape)
	dot_product = torch.dot(mean1, mean2)
	norm1 = torch.norm(mean1)
	norm2 = torch.norm(mean2)
	cosTheta = dot_product/(norm1*norm2)

	g_loss = self.loss_func(cosTheta, self.target_one)

	return g_loss

# make res positive,so that y_fake*res < 0
def loss7(self,gen_data, svm_w, svm_b):

	res = torch.matmul(gen_data, svm_w) +  svm_b
	res = torch.sign(res)
	batch_size = gen_data.shape[0]

	target = torch.ones(batch_size)	
	res = res - target
#	res = torch.linalg.norm(res)  #not work well
	res = torch.sum(res) #not work well either

	return res

#multivariate Gaussian model
def loss8(self,gen_data,ref_data):

	mu1,sigma1 = self.calc_Gaussian_model(gen_data)
	mu2,sigma2 = self.calc_Gaussian_model(ref_data)

	#check if the covariance mat is positive
	self.positive_definite_mat_check(sigma1)
	self.positive_definite_mat_check(sigma2)
	#end of check
	
	pdf_values1 = self.calc_PDF(gen_data,mu1,sigma1)
	pdf_values2 = self.calc_PDF(ref_data,mu2,sigma2)
	
	diff = pdf_values1 - pdf_values2
	return torch.linalg.norm(diff)

#compare the cov_mat
def loss9(self,gen_data, ref_data):
#	if self.model_flag==203:  #lqnote may be modifed later
#		gen_data = gen_data.squeeze(1)
#		ref_data = ref_data.squeeze(1)


	cov_mat1 = torch.cov(gen_data.T)
	cov_mat2 = torch.cov(ref_data.T)

	diff = cov_mat1 - cov_mat2
	diff = torch.linalg.norm(diff)

	return diff
#calc the distance btw centers
def loss10(self,gen_data, ref_data):
#	if self.model_flag==203:  #lqnote may be modifed later
#		gen_data = gen_data.squeeze(1)
#		ref_data = ref_data.squeeze(1)


	center1 = torch.mean(gen_data, dim=0)
	center2 = torch.mean(ref_data, dim=0)
	diff    = center1 - center2
	diff    = torch.linalg.norm(diff)
	return diff	



if __name__=="__main__":
	shift_value=379
	energy_tensor, shiftE_tensor,dipole_tensor, geom_list,Nconfig, atoms = load_data(shift_value)	
	NAtom = 10
	
	GEDlist = setup_tensor_list(shiftE_tensor, dipole_tensor,geom_list,NAtom)

	g_p, args = f02_main()
	
	batch_size = g_p.batch_size
	dataloader = setup_dataloader(GEDlist, batch_size)
	generator = Generator(g_p)

