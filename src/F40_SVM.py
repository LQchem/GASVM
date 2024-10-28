import numpy as np
import torch
import pandas
import matplotlib.pyplot as plt
from F01_loaddata import printmat,load_data
from F02_parameters import *
import random
import copy

class LinearSVM(torch.nn.Module):
	def __init__(self,inp_dim):
		super(LinearSVM, self).__init__()
		self.weight = torch.nn.Parameter(torch.rand(inp_dim), requires_grad=True)
		self.bias = torch.nn.Parameter(torch.rand(1), requires_grad=True)

		self.inp_dim = inp_dim
		self.Nlayer=5
		self.Normalizer = nn.BatchNorm1d
		self.activation_func =  nn.LeakyReLU()#nn.ReLU6() #nn.Softmax()#nn.Sigmoid() #nn.ReLU() #nn.LeakyReLU()

		self.model = self.SVM_model()
		
	def SVM_model(self):
		module_list=[]
		inp_dim = self.inp_dim

		for i in range(self.Nlayer):
			module_list.append(nn.Linear(inp_dim, inp_dim))
			module_list.append(self.Normalizer(inp_dim))
			module_list.append(self.activation_func)

		model = nn.Sequential(*module_list)
		return model

	def forward(self, x):
		#----
		#if use multilayer svm
		#x = self.model(x)
		#----
		return torch.matmul(x, self.weight) + self.bias

	def plot_progress(self,data,title):
		df = pandas.DataFrame(data,columns=['loss'])
		df.plot(marker='.',grid=True,title=title)
		plt.savefig("LinearSVM_loss.png")
		plt.show()



def generate_random(size):
    random_data = torch.rand(size)
    random_data.requires_grad_(True)
    return random_data


if __name__=="__main__":
	dim         = 10

	svm2 = LinearSVM(dim)

	fake_data = generate_random(dim)
	real_data = torch.arange(dim)

	yreal = torch.ones(dim)
	yfake = -yreal
	
	ylabel = torch.cat((yreal,yfake))
	print(ylabel)

	optimizer = torch.optim.SGD([svm2.weight, svm2.bias], lr=0.01)

	fake_data = torch.vstack([fake_data,fake_data])
	print(fake_data.shape)

	data = torch.cat([fake_data, fake_data])
	print(data.shape)
	res = svm2(data)
	print(res)
	print("---")
	print(res.shape)

#	C=0.1
#	for epoch in range(500):
#		y=ylabel[i]
#		optimizer.zero_grad()
#		loss = torch.max(torch.tensor(0),1-y*svm2(data))
#		loss += C*torch.norm(svm2.weight)**2
##		print(loss)
#		svm_loss.append(loss.item())
#		loss.backward()
#		optimizer.step()


#	svm2.plot_progress(svm_loss,"svm_loss")
#	print(len(GE_list))
#	print(GE_list[0].shape)
	



	



#w, b, losses = svm.fit(X_train, y_train)

#prediction = svm.predict(X_test)















