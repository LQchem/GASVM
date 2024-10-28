import numpy as np
import torch
import pandas
import matplotlib.pyplot as plt
from F01_loaddata import printmat,load_data
from F02_parameters import *
import random

class Discriminator(nn.Module):
	def __init__(self,para):
		super().__init__()
		self.info = para
		self.model           = self.D_model()
		self.optimiser       = para.optimizer(self.parameters(),lr=self.info.lr)
		self.counter  = 0
		self.progress = []
#		print("Discriminator model")
#		print(self.model)

	def D_model(self):
		info            = self.info
		model_flag      = info.model_flag
		Nlayer          = info.Nlayer
		NAtom           = 10
		increment       = info.increment
		activation_func = info.activation_func
		batch_size      = info.batch_size
		Normalizer      = self.info.Normalization
		inp_dim         = NAtom*3+1 #xya + E
		self.inp_dim    = inp_dim

		if model_flag==102:
			module_list=[]
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim+increment))
				module_list.append(activation_func)
				module_list.append(Normalizer(inp_dim+increment))
				inp_dim+=increment
			
			for i in range(Nlayer):
				module_list.append(nn.Linear(inp_dim,inp_dim-increment))
				module_list.append(activation_func)
				module_list.append(Normalizer(inp_dim-increment))
				inp_dim-=increment
			
			module_list.append(nn.Linear(inp_dim,1))
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


def generate_random(size):
    random_data = torch.rand(size)
    random_data.requires_grad_(True)
    return random_data



if __name__=="__main__":


	Energy_list, Geom_list, GE_list = load_data(print_flag=0)
	g_p, d_p,args = f02_main()

	discriminator = Discriminator(d_p)
	inp_dim = discriminator.inp_dim
#	rand_inp = generate_random(inp_dim)
#	real_targets = torch.ones_like(rand_inp)
#	fake_targets = torch.zeros_like(rand_inp)

	real_targets = torch.FloatTensor([1])
	fake_targets = torch.FloatTensor([0])


	discriminator_loss=[]

	dataset_size=100
#	Nconfig = random.randint(start_value,len(GE_list))
	subset = random.sample(GE_list,dataset_size)
	Epoch=1000
	for epoch in range(Epoch):
		random.shuffle(subset)
		i=0
		for data in subset:
			discriminator.optimiser.zero_grad()

			real_data = ConvertTensor(data,False)
			d_res     = discriminator.forward(real_data)
			real_loss = discriminator.info.loss_func(d_res, real_targets)

	
			fake_data = generate_random(inp_dim)
			d_res     = discriminator.forward(fake_data)
			fake_loss = discriminator.info.loss_func(d_res,fake_targets)

			d_loss = (real_loss + fake_loss)*0.5
			d_loss.backward()
			discriminator.optimiser.step()
			discriminator_loss.append(d_loss.item())
			i+=1
		print(
                 "[Epoch %d/%d] [Batch %d/%d] [D loss: %f]  "
                 % (epoch, Epoch, i,len(subset), d_loss.item(),  ))

	discriminator.plot_progress(discriminator_loss,"d_loss")

