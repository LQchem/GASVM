import numpy as np
import torch
import pandas
import torchvision
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

from F01_loaddata import *
from F02_parameters import *
from F12_G_models import *
from F16_G_loss import *
from F21_resnet import *
from F22_resnet1d import *
class Generator(nn.Module):
	def __init__(self,args):
		super().__init__()
		
		self.args = self.load_para(args)
		print_generator(self)


		self.model = self.set_generator_model()

		self.target_zero = torch.FloatTensor([0])
		self.target_one = torch.FloatTensor([1])

		self.loss  = self.set_loss_model()
	
		self.optimiser = self.optimizer(self.parameters(),lr=self.lr)
		

		print(self.model)

	def set_generator_model(self):
		if self.model_flag == 101: #FC
			return Linear_Model1(self)
		if self.model_flag == 102: #simple one layer FC
			return Linear_Model2(self)
		if self.model_flag == 103: #simple one layer FC
			return Linear_Model3(self)


		if self.model_flag == 111: #cnn
			print("use 1d cnn")
			return convo1d_model1(self) 
		if self.model_flag == 121: #cnn use resnet 1d
			print("use resnet 1d cnn")
			return ResNet1D(self) 

		if self.model_flag == 201: #resnet for mnist
			return ResNet18(self)
		if self.model_flag == 251: #resnet for my data
			return ResNet18(self)

	def set_loss_model(self):
		loss = self.loss1 = loss1
		loss = self.loss2 = loss2
		loss = self.loss3 = loss3
		loss = self.loss5 = loss5
		loss = self.loss6 = loss6
		loss = self.loss7 = loss7 
		loss = self.loss8 = loss8
		loss = self.loss9 = loss9
		loss = self.loss10 = loss10


		if self.loss_model==1: #cluster analysis
			print("use cluster analysis to calculate g loss")
			return loss1
		if self.loss_model==2: #JS div
			print("use JS divergence to calculate g loss")
			return loss2
		if self.loss_model==3: #KV div,not working 
			pass
		if self.loss_model==4: # quantile distribution
			print("use quantule distribution to calculate g loss")
			return loss4
		if self.loss_model==5: #fit vector, need to extract y, supervised learning
			print("use fit vector to calcualte g loss")
			return loss5
		if self.loss_model==6: # vector angle, btw generated vector and real vector
			print("use vector angle to calculate g loss")
			return  loss6
		if self.loss_model==7:# svm w and b
			print("loss7, use svm weight and bias to calcualte g loss")
			return loss7
		if self.loss_model==8:#multivariate Gaussian model
			print("loss8, Gaussian multivariate model")
			return loss8
		if self.loss_model==9:#covariance matrix
			print("loss9, covariance matrix")
			return loss9
		if self.loss_model==10:#kmeans,center diff
			print("loss10, kmeans center diff")
			return loss10

		return loss




	def load_para(self,args):
		for attr_name, attr_value in vars(args).items():
			setattr(self,attr_name,attr_value)
		return args
		
	def forward(self,inp_tensor):

		if self.model_flag == 111: #lqnote might rewritte this part if conv1d is decided to use
			out = self.model(inp_tensor) #the unsqueeze is done in F72
			#cnn_output_size = calc_convo1D_Nfeature(self.out_Nfeature,self.kernel_size, self.stride, self.padding, self.dilation, self.pool_kernel_size, self.pool_stride)			
			flatten_out = out.view(self.batch_size,-1)
			cnn_output_size = flatten_out.shape[1]
			out = nn.Linear(cnn_output_size, self.out_Nfeature)(flatten_out)
	#		print(out.shape)
		elif self.model_flag == 201:
#			print("aaaaa", inp_tensor.shape)
			out = self.model(inp_tensor)
		
		else:
			out = self.model(inp_tensor)
		return out

	def plot_progress(self,data,title):
		df = pandas.DataFrame(data,columns=['loss'])
		df.plot(marker='.',grid=True,title=title)
		plt.savefig("generator_loss.png")
		plt.show()


	#use the Gaussian multivariable model to generate random input
	def generate_GMM_random(self,ref_data):
		mean_vec = torch.mean(ref_data, axis=0)
		std_vec  = torch.std(ref_data, axis=0)	 #calc the std along each vector
		batch_size, Nfeature = ref_data.shape

		vec_list = [] 

		for i in range(Nfeature):
			#generate gaussian distribution
			#new_vec = torch.normal(mean=mean_vec[i], std=std_vec[i], size=(batch_size, 1))

			#generate uniform distribution
			new_vec = torch.rand((batch_size, 1))*std_vec[i] + mean_vec[i]


			vec_list.append(new_vec)		

		gen_data = torch.hstack(vec_list)
		return gen_data

	def generate_random(self,batch_size=-1, target_dim=-1):

		if batch_size < 0:
			batch_size = self.batch_size
		if target_dim < 0:
			target_dim = self.target_dim

		if self.model_flag==201:#resnet for mnist
			random_data = torch.rand(batch_size,1, target_dim,target_dim)

		elif self.model_flag==111: #resnet
			random_data = torch.rand(batch_size,1, target_dim * target_dim) 

		elif self.model_flag==121: #conv1d
			random_data = torch.rand(batch_size,1, target_dim )

		elif self.model_flag==251: #resnet for my data	
			random_data = torch.rand(batch_size,3, self.NAtom,3)

		else:
			random_data = torch.rand(batch_size, target_dim)
#		random_data.requires_grad_(True)
		return random_data

	def generate_resnet_random(self,batch_size, NAtom):#not in use
		random_data = torch.rand(batch_size, 3,NAtom, 3)
#		random_data.requires_grad_(True)
		return random_data


	#---------

	def positive_definite_mat_check(self,mat):
		from torch.linalg import eigvals
		eigenvalues = eigvals(mat)
		print(eigenvalues)
		if torch.all(eigenvalues>0):
			pass
		else:
			print("matrix is not positive denfinite")



	def calc_Gaussian_model(self,data):
		self.mean_vec = torch.mean(data,axis=0)
		self.cov_mat  = torch.cov(data.T) #data is of shape (Nsampe, Nfeature), so need to do the transformation

	#   if calc mean and cov for the whole datalist:
	#	alldata = torch.vstack(my_datalist)
	#	mu2,sigm2 = feature.calc_Gaussian_model(alldata)

		return self.mean_vec, self.cov_mat

	def calc_PDF(self,data, mean_vec, cov_mat):
		'''	calc probability density function
			mean_vec: the mean vector
			cov_mat: the covariance matrix
		'''	
#		cov_mat = cov_mat + 1e-8*torch.eye(cov_mat.size(0)) #add a small number on the diagonal terms 
		mvn = MultivariateNormal(loc=mean_vec, covariance_matrix = cov_mat)
		pdf_values = mvn.log_prob(data).exp()
		return pdf_values


	#---------



if __name__=="__main__":
#	shift_value=379
#	energy_tensor, shiftE_tensor,dipole_tensor, geom_list,Nconfig, atoms = load_data(shift_value)	
#	NAtom = 10
	
#	GEDlist = setup_tensor_list(shiftE_tensor, dipole_tensor,geom_list,NAtom)

	g_p, args = f02_main()
	

	generator = Generator(g_p)

#	random_inp = generator.generate_resnet_random(g_p.batch_size, g_p.NAtom)
	random_inp = generator.generate_random(g_p.batch_size, g_p.target_dim)
	print("random_inp shape",random_inp.shape)

	if generator.model_flag==111:
	#	random_inp = random_inp.unsqueeze(1)
		out = generator.forward(random_inp)
		print("random inp shape after unsquezze", random_inp.shape)

	elif generator.model_flag==201:
	#	random_inp = random_inp.unsqueeze(1)
		out = generator.forward(random_inp)
	else:
		out = generator.forward(random_inp)

	print(out.shape)
