import numpy as np
import os
import copy
import torch
import csv
import pandas as pd
			
def Prep_file(outfile, datfile):
	f_geom = open(outfile,"w")
	for filename in datfile:
		f=open(filename)	
		lines = f.readlines()
		for line in lines:
			f_geom.write(line)
		f.close()




def load_data():
	files = os.listdir("./")

	datfile=[]
	txtfile=[]
	for f in files:
		if f.endswith(".dat"):
			datfile.append(f)


	datfile.sort() #geom

	outfile1 = "all_dipole.csv"
	Prep_file(outfile1,datfile)



if __name__=="__main__":
	#flatten the geom
	#and shift the energy
	print("prep data files")
	load_data()
