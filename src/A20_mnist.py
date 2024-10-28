import torch
import pandas as pd
import numpy as np
def ConvertTensor(vec,grad_flag=True):
	return torch.tensor(vec,dtype=torch.float32,requires_grad=grad_flag)


def load_mnist():
	filename = "../csvfiles/mnsit60000.csv"
	data = pd.read_csv(filename,header=None)
	data = np.mat(data)

	data_list = []
	for i in range(data.shape[0]):
		row=np.ravel(data[i,:])
#		row=np.ravel(row)		
		row = ConvertTensor(row,False)
		data_list.append(row)

	return data, data_list


