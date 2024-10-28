import numpy as np
import os
import copy
import torch
import csv
import pandas as pd
from torch.utils.data import Dataset

from F01_loaddata import *

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
	




if __name__=="__main__":

	shift_value=379
	energy_tensor, shiftE_tensor,dipole_tensor, geom_list,Nconfig, atoms = load_data(shift_value)
	NAtom=10
	datalist = setup_tensor_list(shiftE_tensor, dipole_tensor,geom_list,NAtom)

	check(datalist)
#	
	
