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

		if f.endswith(".txt"):
			txtfile.append(f)

	datfile.sort() #geom
	txtfile.sort() #energy

	outfile1 = "all_geom.csv"
	Prep_file(outfile1,datfile)

	outfile2 = "all_energy.csv"
	Prep_file(outfile2,txtfile)


if __name__=="__main__":
	#flatten the geom
	#and shift the energy
	print("prep data files")
	load_data()
