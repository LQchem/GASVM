import numpy as np
import os
import copy
import torch
import csv
import pandas as pd
from torch.utils.data import Dataset
torch.set_printoptions(precision=8)

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
	

	ori_geom = np.loadtxt("geom.csv")
	deltaGeom = np.linalg.norm(Garray-ori_geom)
	print("deltaG", deltaGeom)	

	ori_E = np.loadtxt("energy.csv")
	deltaE =np.linalg.norm( ori_E-Earray )
	print("deltaE", deltaE)
	
	ori_d = np.loadtxt("dipole.csv")
	deltaD =np.linalg.norm( ori_d-Darray )
	print("deltaD", deltaD)
	

def load_dipole(filename,dim):
	f=open(filename)
	lines = f.readlines()
	dipole = torch.zeros([dim,3],dtype=torch.float64) 
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
	energy   = torch.zeros([Nconfig,1],dtype=torch.float64)
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
#	geom = torch.zeros([dim,3],dtype=torch.float64)
	geom = torch.zeros([NAtom,3],dtype=torch.float64)
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
			atoms=[]
			geom = torch.zeros([NAtom,3],dtype=torch.float64)

	return geom_list,atoms


def load_data(shift_value=379):
	print("load data")
	dipole_file = "all_dipole.csv"
	energy_file = "all_energy.csv"
	geom_file   = "all_geom.csv"

	energy_tensor,Nconfig, shift_E = load_energy(energy_file,shift_value)
	dipole_tensor                  = load_dipole(dipole_file, Nconfig)
	geom_list,atoms                = load_geom(geom_file, Nconfig)

	return energy_tensor, shift_E, dipole_tensor, geom_list,Nconfig, atoms

def setup_tensor_list(Energy,Dipole,GeomList,NAtom):
	print("setup data list")
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

class myDataset(Dataset):
	def __init__(self,data):
		self.data = data

	def __len__(self):
		pass		



if __name__=="__main__":
	shift_value=379
	energy_tensor, shiftE_tensor,dipole_tensor, geom_list,Nconfig, atoms = load_data(shift_value)
	NAtom=10

	datalist = setup_tensor_list(shiftE_tensor, dipole_tensor,geom_list,NAtom)
	check(datalist)	
