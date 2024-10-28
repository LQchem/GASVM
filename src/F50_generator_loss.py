import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
#from scipy.special import kl_div
from scipy import stats

class LinearFit(nn.Module):
	def __init__(self,inp_dim):
		super(LinearFit, self).__init__()
		self.model     = nn.Linear(inp_dim-1, 1)
		self.loss_func = nn.MSELoss()
		self.lr        = 0.001
		self.optimizer = optim.SGD(self.model.parameters(),lr=self.lr)
	
	def load_data(self, data):
		self.X = data[:,0:-1]
		self.Y = data[:,-1]

	def fit(self, num_epochs=100):
		for epoch in range(num_epochs):
			y_pred = self.model(self.X)
			loss = self.loss_func(y_pred, self.Y)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		weight = self.model.weight
		bias   = self.model.bias
		return weight, bias



def Kolmogorov_Smirnov_divergence(data1, data2):
	"""
	计算两个数据集之间的KS散度。
	
	参数:
	data1, data2: 两个数据列表或数组，代表要比较的两个数据集的观测值。
	
	返回:
	ks_statistic: KS统计量，表示两个分布的最大差异。
	p_value: p值，用于判断两个数据集的分布是否有显著差异。
	"""
	# 将数据转换为numpy数组以提高计算效率（如果输入不是数组的话）

	data1 = data1.detach().numpy()
	data2 = data2.detach().numpy()


	data1 = np.average(data1,axis=0)
	data2 = np.average(data2,axis=0)
	# 使用SciPy的ks_2samp函数计算KS统计量和p值
	ks_statistic, p_value = stats.ks_2samp(data1, data2)
	
	return p_value

def jensen_shannon_divergence(p, q, eps=1e-10):
	"""
	Compute the Jensen-Shannon Divergence between two probability distributions.
	
	Args:
	    p (torch.Tensor): First probability distribution.
	    q (torch.Tensor): Second probability distribution.
	    eps (float): Small value to avoid division by zero or log(0).
	    
	Returns:
	    torch.Tensor: Jensen-Shannon Divergence.
	"""
	# transform p and q into [0,1] 

	#----
	#attempt1	
#	p = torch.sigmoid(p)
#	q = torch.sigmoid(q)	
	#----

	p = (p-torch.min(p)) / (torch.max(p)-torch.min(p))
	q = (q-torch.min(q)) / (torch.max(q)-torch.min(q)) 

	# Calculate the average distribution
	m = 0.5 * (p + q)
	
	# Compute the divergences using the log-sum-exp trick to avoid numerical instability
	divergence_p = torch.sum(p * torch.log((2 * p + eps) / (m + eps)))
	divergence_q = torch.sum(q * torch.log((2 * q + eps) / (m + eps)))
	
	# Return the average divergence
	js_div = 0.5 * (divergence_p + divergence_q)
	
	return js_div





if __name__=="__main__":
	# Example usage
	p = torch.tensor([-1, 2, 3])
	q = torch.tensor([4, 2, 6])
	
	jsd = jensen_shannon_divergence(p, q)
	print(f"Jensen-Shannon Divergence: {jsd.item()}")
#	ksd = Kolmogorov_Smirnov_divergence(p, q)
#	print(ksd)
	test=LinearFit(123)

