import numpy as np
import os
import copy
import torch
import torch.nn as nn
import csv
import pandas as pd
from A20_mnist import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, Normalizer

from torch.utils.data import Dataset
torch.set_printoptions(precision=8)

def scaling_data(array,flag=1):
	if flag==1: #Min-Max scaling, map to [0,1]

#		mm_scaler = MinMaxScaler()
#		s_arry    = mm_scaler.fit_transform(array)

		xmin = np.min(array)
		xmax = np.max(array)
		s_array = (array - xmin) / (xmax - xmin)


	if flag==2: #Standardization, Gaussian distribution with  mu and sigma
		std_scaler = StandardScaler()
		s_array = std_scaler.fit_transform(array)

	if flag==3: #Max abs scaling,map to [-1,1]
		ma_scaler = MaxAbsScaler()
		s_array   = ma_scaler.fit_transform(array)		
	if flag==4: #Robust scaling
		rb_scaler = RobustScaler()
		s_array   = rb_scaler.fit_transform(array)
	
	return s_array


def ConvertTensor(vec,grad_flag=True):
	return torch.tensor(vec,dtype=torch.float32,requires_grad=grad_flag)



def printmat(mat):
        r,c=mat.shape
        for i in range(r):
                for j in range(c):
                        print("%10.6f" %mat[i,j],"   ",end='')
                print()

def check(datalist):
	#check the tensor list 
	g_list = []
	e_list = []
	d_list = []
	for GED in datalist:
		geom   = GED[0,:,:].numpy()
		energy = GED[1,0,0].numpy()-shift_value
		dipole = GED[2,0,:].numpy()
#		print(energy)
		g_list.append(geom)
		e_list.append(energy)
		d_list.append(dipole)

	Garray = np.vstack(g_list)
	Earray = np.array(e_list)
	Darray = np.vstack(d_list)
	

	ori_geom = np.loadtxt("../check/geom.csv")
	deltaGeom = np.linalg.norm(Garray-ori_geom)
	print("deltaG", deltaGeom)	

	ori_E = np.loadtxt("../check/energy.csv")
	deltaE =np.linalg.norm( ori_E-Earray )
	print("deltaE", deltaE)
	
	ori_d = np.loadtxt("../check/dipole.csv")
	deltaD =np.linalg.norm( ori_d-Darray )
	print("deltaD", deltaD)
	

def load_dipole(filename,dim):
	f=open(filename)
	lines = f.readlines()
	dipole = torch.zeros([dim,3],dtype=torch.float32) 
	count=0
	for line in lines:
		line = line.strip()
		line = line.split()	
		dx   = float(line[-3])
		dy   = float(line[-2])
		dz   = float(line[-1])
		dipole[count,0] = dx
		dipole[count,1] = dy
		dipole[count,2] = dz
		count+=1
	return dipole


def load_energy(filename,shift_value):
	f = open(filename)
	lines    = f.readlines()
	Nconfig  = len(lines)
	energy   = torch.zeros([Nconfig,1],dtype=torch.float32)
	count    = 0
	for line in lines:
		line = line.strip()
		line = line.split()
		Eng  = float(line[-1])
		energy[count,0] = Eng
		count+=1

	shift_energy = energy+shift_value

	return energy,Nconfig,shift_energy


def load_geom(filename,Nconfig,NAtom=10):
	f = open(filename)
	lines = f.readlines()
	dim = Nconfig*NAtom
#	geom = torch.zeros([dim,3],dtype=torch.float32)
	geom = torch.zeros([NAtom,3],dtype=torch.float32)
	geom_list = []
	count = 0
	atoms=[]
	for line in lines:
		if "point" in line:
			continue
		line = line.strip()
		line = line.split()
		atom,x,y,z = line		
		atoms.append(atom)
		geom[count,0] = float(x)
		geom[count,1] = float(y)
		geom[count,2] = float(z)
		count+=1

		if count==NAtom:
			count=0
			geom_list.append(geom)
			geom = torch.zeros([NAtom,3],dtype=torch.float32)

	return geom_list,atoms[0:NAtom]


def load_data(shift_value=379):
	print("load data")
	dipole_file = "../data/all_dipole.csv"
	energy_file = "../data/all_energy.csv"
	geom_file   = "../data/all_geom.csv"

	energy_tensor,Nconfig, shift_E = load_energy(energy_file,shift_value)
	dipole_tensor                  = load_dipole(dipole_file, Nconfig)
	geom_list,atoms                = load_geom(geom_file, Nconfig)



	return energy_tensor, shift_E, dipole_tensor, geom_list,Nconfig, atoms

#setup the tensor list for CNN, whose element's size is of size (3,NAtom,3), 
def setup_tensor_list(Energy,Dipole,GeomList,NAtom):
	print("setup GED tensor list for CNN, each tensor is of size (3, NAtom, 3)")
	Nconfig = len(GeomList)
	shape   = (NAtom,3)

	ones = torch.ones([NAtom,3])
	datalist=[]

	for i in range(Nconfig):
		energy = Energy[i]
		dipole = Dipole[i]
		geom   = GeomList[i]

		Emat = 	energy*ones
		tmpD = []
		for j in range(NAtom):
			tmpD.append(dipole)
		Dmat = torch.vstack(tmpD)
		
		Etensor = Emat.view(1,*shape)
		Dtensor = Dmat.view(1,*shape)
		Gtensor = geom.view(1,*shape)
	
		GED     = torch.vstack([Gtensor,Etensor, Dtensor])
		datalist.append(GED)
	
	return datalist

#setup the tensor list for Linear NN, whose element's size is of size (3*NAtom+4), 
def setup_tensor_list2(Energy, Dipole, Geom_list):
	print("setup GED tensor list for FC NN, each tensor is of size (3Natom+1+3)")
	dim = len(Geom_list)
	GED_list=[]
	scaled_GED_list=[]
	for i in range(dim):
		flat_geom = list(Geom_list[i].flatten())
		energy    = Energy[i].detach().numpy()
		dipole    = Dipole[i].detach().numpy()
		flat_geom.extend(energy)
		flat_geom.extend(dipole)

		#----
		#do the scaling
		array = np.array(flat_geom)
		s_array = scaling_data(array,flag=1) 
		#----

		GED_list.append(array)
		scaled_GED_list.append(s_array)

	return GED_list,scaled_GED_list



class myDataset(Dataset):
	def __init__(self,datalist):
		self.datalist = datalist

	def __len__(self):
		return len(self.datalist)

	def __getitem__(self,index):
		return self.datalist[index]


def setup_dataloader(datalist,batch_size,shuffle=True):
#	if flag=='c':
	_dataset_ = myDataset(datalist)
	dataloader = torch.utils.data.DataLoader(_dataset_,batch_size=batch_size,shuffle=shuffle,drop_last=True)
	

	return dataloader

def layer_normalization(tensor,flag):
	if flag=='g' or flag=='d':
		layer_norm = nn.LayerNorm(3)
		output = layer_norm(tensor)

	if flag=='e':
		shape = tensor.shape
		layer_norm = nn.LayerNorm(shape[1])
		output = layer_norm(tensor)	

	return output

if __name__=="__main__":
	flag=1
	if flag==1:

		shift_value=379
		#energy_tensor of shape N*1,dipole_tensor of shape N*3, N is the total number of samples
		energy_tensor, shiftE_tensor,dipole_tensor, geom_list,Nconfig, atoms = load_data(shift_value) #geom_list members in tensor type of (N,3) shape
		NAtom=10
	
		#if need to do some scaling transformation, do it here
		#do_some_transormationi
		pre_normalization=1
		if pre_normalization:
			print("layer normalization")
			layer_normalized_geom=[]
			for g in geom_list:
				lg = layer_normalization(g,"g")
				layer_normalized_geom.append(lg)
			layer_normalized_energy = layer_normalization(energy_tensor,flag='e')
			layer_normalized_dipole = layer_normalization(dipole_tensor,flag='d')
	
			print("finish layer normalization")
	
		#setup the tensor list for CNN, whose element's size is of size (3,NAtom,3), 
		#GEDlist = setup_tensor_list(layer_normalized_energy, layer_normalized_dipole, layer_normalized_geom,NAtom)
		GEDlist = setup_tensor_list(shiftE_tensor, dipole_tensor,geom_list,NAtom)
	
		#setup the tensor list for LNN, whose element's size is of (NAtom*3+4) 
		GEDlist2,scaled_GED_list2 = setup_tensor_list2(shiftE_tensor, dipole_tensor,geom_list)
		
		
		batch_size=80
		dataloader_c = setup_dataloader(GEDlist, batch_size) #for CNN
		dataloader_l = setup_dataloader(GEDlist2,batch_size) #for linear NN


	if flag==2:
		mnist_data,data_list = load_mnist()
		batch_size = 64
		dataloader_m = setup_dataloader(data_list, batch_size)

		print(data_list[0].shape)

