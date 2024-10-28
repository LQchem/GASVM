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

#	ncenter=6000
#	centers = np.loadtxt(f"mnist_centers_{ncenter}.csv")
#	print(kmeans_file)
	centers = np.loadtxt(kmeans_file)
	centers = ConvertTensor(centers, False)

	accum_dist = 0
#	if gen_p is of type list
#	gen_point_list = gen_data
#	for gen_p in gen_point_list:
#		accum_dist += find_closet_center(gen_p, centers)

#	if gen_p is of type tensor
	for i in range(gen_data.shape[0]):
		accum_dist = find_closet_center(gen_data[i,:],centers)


	accum_dist.requires_grad_()
	return 	accum_dist





class Generator(nn.Module):
	def __init__(self,para):
		super().__init__()
		self.info = para
		self.batch_size      = para.batch_size
		self.target_zero     = torch.FloatTensor([0])
		self.target_one      = torch.FloatTensor([1])

		self.inp_dim       = self.info.inp_dim
		self.linear_fit1   = LinearFit(self.inp_dim) 
		self.linear_fit2   = LinearFit(self.inp_dim) 
		self.kmeans_file   = para.kmeans_file 	
		#print("asfasdfasdf")
		#print(self.kmeans_file)

		self.model           = self.G_model()
		self.optimiser       = para.optimizer(self.parameters(),lr=self.info.lr)
		self.counter  = 0
		self.progress = []

#		self.cluster_anal_threshold = torch.FloatTensor([2.0]) #the threshold to tell difference for cluster analysis
		print("Generator model")
		print(self.model)

	def G_model(self):
		info            = self.info
		model_flag      = info.model_flag
		Nlayer          = info.Nlayer
		NAtom           = 10
		increment1      = info.increment
		increment2      = info.increment2
		activation_func = info.activation_func
		activation_func2 = info.activation_func2
		
		Normalizer      = self.info.Normalization
#		inp_dim         = NAtom*3+1 #xya + E
		inp_dim         = self.inp_dim
		latent_dim      = self.info.latent_dim


		if model_flag==101: #lqnote transform/scale generated data
			module_list=[]
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim+increment1))
				module_list.append(activation_func)
				module_list.append(Normalizer(inp_dim+increment1))
				inp_dim+=increment1
			
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim-increment1))
				module_list.append(activation_func)
				module_list.append(Normalizer(inp_dim-increment1))
				inp_dim-=increment1
		

			module_list.append(activation_func2)
			model = nn.Sequential(*module_list)

		if model_flag==102: #initial attempts
			module_list=[]
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim+increment1))
				module_list.append(activation_func)
				module_list.append(Normalizer(inp_dim+increment1))
				inp_dim+=increment1
			
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim-increment1))
				module_list.append(activation_func)
				module_list.append(Normalizer(inp_dim-increment1))
				inp_dim-=increment1
			
			module_list.append(nn.Linear(inp_dim,inp_dim))
			module_list.append(activation_func)
			
			model = nn.Sequential(*module_list)

		if model_flag==104:
			module_list=[]
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim+increment1))
				module_list.append(activation_func)
				module_list.append(Normalizer(inp_dim+increment1))
				inp_dim+=increment1
			
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim-increment1))
				module_list.append(activation_func)
				module_list.append(Normalizer(inp_dim-increment1))
				inp_dim-=increment1
			
			module_list.append(nn.Linear(inp_dim,inp_dim))
			module_list.append(activation_func2)
			
			model = nn.Sequential(*module_list)


		if model_flag==201: #use resnet

	
			module_list=[]
			for i in range(Nlayer):
				module_list.append(nn.Linear(latent_dim,latent_dim + increment1 + increment2))
				module_list.append(activation_func)
				module_list.append(Normalizer(latent_dim + increment1 + increment2))
				latent_dim+= (increment1 + increment2)
		
#			resnet = torchvision.model(resnet18(pretrained=True) #True if use pretrained weights
#			module_list.append(resnet)
#			inp_dim2 = resnet_Nfeature = resnet.fc.in_features
				
			for i in range(Nlayer):
				module_list.append(nn.Linear(latent_dim,latent_dim-increment1-increment2))
				module_list.append(activation_func)
				module_list.append(Normalizer(latent_dim-increment1-increment2))
				latent_dim -= (increment1+increment2)
			
			module_list.append(nn.Linear(latent_dim,self.inp_dim))
			module_list.append(activation_func)
			
			model = nn.Sequential(*module_list)


		return model

	
	def forward(self,inp_tensor):
		out = self.model(inp_tensor)
		return out


	def train(self,inputs,targets):
		self.optimiser.zero_grad()
		outputs = self.forward(inputs)
		
		loss = self.info.loss_func(outputs,targets)
		self.counter+=1
		if self.counter%10 == 0:
			self.progress.append(loss.item())
		if self.counter%10000 ==0:
			print("counter = ", self.counter)

		self.optimiser.step()
		return loss

	def plot_progress(self,data,title):
		df = pandas.DataFrame(data,columns=['loss'])
		df.plot(marker='.',grid=True,title=title)
		plt.savefig("discriminator_loss.png")
		plt.show()


	def generate_random(self,batch_size, inp_dim):
		random_data = torch.rand(batch_size, inp_dim)
#		random_data.requires_grad_(True)
		return random_data


	#cluster analysis
	def loss1(self,gen_data, threshold=torch.FloatTensor([2.0]) ):
		accum_dist = cluster_analysis(gen_data,self.kmeans_file)
		g_loss = self.info.loss_func(accum_dist, threshold)	
		return g_loss

	#JS divergence
	def loss2(self, gen_data, ref_tensor):
		js_div = jensen_shannon_divergence(gen_data, ref_tensor)
		g_loss = self.info.loss_func(js_div, self.target_zero)
		return g_loss

	#KS divergence #code not working, and may not be useful
	def loss3(self, gen_data, ref_tensor):
		p_value = Kolmogorov_Smirnov_divergence(gen_data, ref_tensor)
		g_loss = self.info.loss_func(p_value, self.target_one)
		return g_loss

	#quantile	
	def loss4(self,gen_data, ref_tensor):
		q1 = torch.quantile(gen_data, 0.25,dim=0)
		q2 = torch.quantile(gen_data, 0.50,dim=0)
		q3 = torch.quantile(gen_data, 0.75,dim=0)

		p1 = torch.quantile(ref_tensor, 0.25,dim=0)
		p2 = torch.quantile(ref_tensor, 0.50,dim=0)
		p3 = torch.quantile(ref_tensor, 0.75,dim=0)

		l1 = self.info.loss_func(p1,q1)
		l2 = self.info.loss_func(p2,q2)
		l3 = self.info.loss_func(p3,q3)

		return l1+l2+l3


	#overlap between two fit vector
	def loss5(self,gen_data, ref_tensor,maxit=100):
		self.linear_fit1.load_data(ref_tensor.detach())
		self.linear_fit2.load_data(gen_data.detach())

		# Fit the models but don't compute gradients in this custom loss function
	#	with torch.no_grad():
		w1, b1 = self.linear_fit1.fit(num_epochs = maxit) 
		w2, b2 = self.linear_fit2.fit(num_epochs = maxit)

		g_loss = self.info.loss_func(w1, w2) + self.info.loss_func(b1,b2)
		return g_loss

	#vector angle
	def loss6(self, gen_data, ref_tensor):
		mean1 = torch.mean(gen_data, dim=0)
		mean2 = torch.mean(ref_tensor, dim=0)

#		print(mean1.shape)
		dot_product = torch.dot(mean1, mean2)
		norm1 = torch.norm(mean1)
		norm2 = torch.norm(mean2)
		cosTheta = dot_product/(norm1*norm2)

		g_loss = self.info.loss_func(cosTheta, self.target_one)

		return g_loss


if __name__=="__main__":
	shift_value=379
	energy_tensor, shiftE_tensor,dipole_tensor, geom_list,Nconfig, atoms = load_data(shift_value)	
	NAtom = 10
	
	GEDlist = setup_tensor_list(shiftE_tensor, dipole_tensor,geom_list,NAtom)

	g_p, args = f02_main()
	
	batch_size = g_p.batch_size
	dataloader = setup_dataloader(GEDlist, batch_size)
	generator = Generator(g_p)

