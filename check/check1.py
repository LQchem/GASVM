import numpy as np
import os
import copy
import torch
import csv
import pandas as pd

torch.set_printoptions(precision=8)
def ConvertTensor(vec,grad_flag=True):
	return torch.tensor(vec,dtype=torch.float32,requires_grad=grad_flag)



def printmat(mat):
        r,c=mat.shape
        for i in range(r):
                for j in range(c):
                        print("%10.6f" %mat[i,j],"   ",end='')
                print()



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
	geom = torch.zeros([dim,3],dtype=torch.float64)

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


	print("write flatten geom")
	fw=open("flat_geom.csv","w")
	flat_geom = geom.view(-1,30)
	row,col = flat_geom.shape
	for i in range(row):
		line=''		
		for j in range(col):
			line += str(flat_geom[i,j].detach().numpy())+"  "
		line+="\n"
		fw.write(line)
	

	return geom,atoms


def load_data(shift_value=379):
	dipole_file = "all_dipole.csv"
	energy_file = "all_energy.csv"
	geom_file   = "all_geom.csv"

	energy_tensor,Nconfig, shift_E = load_energy(energy_file,shift_value)
	dipole_tensor                  = load_dipole(dipole_file, Nconfig)
	geom_tensor,atoms              = load_geom(geom_file, Nconfig)

	return energy_tensor, shift_E, dipole_tensor, geom_tensor,Nconfig, atoms

def setup_tensor(Energy,Dipole,Geom,NAtom):

	shape = Geom.shape
	#-----------------
	#match the shape of energy to geom
	ones = torch.ones([NAtom,3])
	emat_list=[]
	for i in range(Energy.shape[0]):
		eng  = Energy[i]
		Emat = ones * eng
		emat_list.append(Emat)
	Estack = torch.vstack(emat_list)
	#-------------

	#----------------
	#match the shape of dipole to geom
	dmat = torch.zeros([NAtom,3])
	dmat_list=[]
	tmp_list=[]
	for i in range(Dipole.shape[0]):
		dipole = Dipole[i]
		for j in range(NAtom):
			tmp_list.append(dipole)
		dmat = torch.vstack(tmp_list)
		tmp_list=[]
		dmat_list.append(dmat)
	Dstack = torch.vstack(dmat_list)
	#---------------

	tensor = torch.zeros([3,*shape],dtype=torch.float64)

	Estack = Estack.view(1,*Estack.shape)
	Dstack = Dstack.view(1,*Dstack.shape)
	Geom   = Geom.view(1,*shape)

	tensor = torch.vstack([Geom, Estack,Dstack])

	return tensor


if __name__=="__main__":
	#attempt1
	#and shift the energy
	shift_value=379
	energy_tensor, shiftE_tensor,dipole_tensor, geom_tensor,Nconfig, atoms = load_data(shift_value)
	NAtom=10
	tensor = setup_tensor(shiftE_tensor, dipole_tensor,geom_tensor,NAtom)

	#check the tensor 
	#check geom
	print(tensor.shape)	
	geom = tensor[0,:,:].numpy()
	Energy = tensor[1,:,:]
	dipole = tensor[2,:,:]

	ori_geom = np.loadtxt("geom.csv")
	deltaGeom = np.linalg.norm(geom-ori_geom)
	print("deltaG", deltaGeom)	

	#check energy
	Elist=[]
	for i in range(0,Energy.shape[0],10):	
		e=Energy[i,0]-379
#		print(e)
		Elist.append(e)
	Earray = np.array(Elist)
	ori_E = np.loadtxt("energy.csv")
	deltaE =np.linalg.norm( ori_E-Earray )
	print("deltaE", deltaE)
	
	#check dipole
	dlist=[]
	for i in range(0,dipole.shape[0],10):	
		d=dipole[i,:]
		dlist.append(d.numpy())
	darray = np.array(dlist)
	ori_d = np.loadtxt("dipole.csv")
	deltaD =np.linalg.norm( ori_d-darray )
	print("deltaD", deltaD)
	







