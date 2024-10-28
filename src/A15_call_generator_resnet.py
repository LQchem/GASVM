import os
import torch
import pandas as pd
from F11_Generator_reformat import *
from F02_parameters import *

def inv_sigmoid(t):
	return torch.log( (1/t)-1 )
def inv_tanh(t):
	return torch.atanh(t)

def make_orca_input(gen_folder,basename,geom,atoms):
	xyzfile = basename+".xyz"
	inpfile = basename+".inp"
	xyzfile = os.path.join(gen_folder, xyzfile)
	inpfile = os.path.join(gen_folder, inpfile)

	#gen xyz file
	NAtom = len(atoms)
	fw = open(xyzfile,"w")
	fw.write(str(NAtom))
	fw.write("\n\n")
	for i in range(NAtom):
		line = f"{atoms[i]}  {geom[i,0]}  {geom[i,1]}  {geom[i,2]}\n"
		fw.write(line)	

	fw.close()	


	#gen inp file
	fw = open(inpfile,"w")
	fw.write("!pal DLPNO-CCSD(T) aug-cc-pvtz aug-cc-pvtz/c rijk aug-cc-pvtz/jk tightpno TightSCF nopop\n")
	fw.write("!angs\n")
	fw.write("%pal nproc 10 end\n")
	fw.write("%MaxCore 6000\n\n")
	fw.write("*xyz 0 1 \n")
	for i in range(NAtom):
		line = f"{atoms[i]}  {geom[i,0]}  {geom[i,1]}  {geom[i,2]}\n"
		fw.write(line)	
	fw.write("*\n")
	fw.close()



if __name__=="__main__":
	'''
	uses F11 generator
	the input feature is of 3 channeld of G E D
	E is shifted
	'''

	pth_file = "generator.pth"
	generator = torch.load(pth_file)


	shift=379	
	
	gen_folder = "Gen_GASVM"
	os.makedirs(gen_folder,exist_ok=True)
	print("generated structrure saved to ", gen_folder)

	atoms=["H","H","H","H","O","O","O","O","C","C"]

#	inp_dim = generator.out_Nfeature	
	dummy_value= 999
	input_tensor = generator.generate_random(generator.batch_size, dummy_value) #if use model 205, the 2nd parameter is not used


	print("input_tensor shape",input_tensor.shape)
	batch_data = generator.forward(input_tensor).detach()
	print("batch_data shape",batch_data.shape)


	#------
	#setup the back transformation function
	g_p, args = f02_main()
	if args.G_activation_func2 ==2:
		back_transformation = inv_sigmoid 
	elif args.G_activation_func2 ==1:
		back_transformation = inv_tanh
	#----

	print(generator.model)

	
	
	for i in range(generator.batch_size):
		gen_data = batch_data[i,:]
#		print(gen_data)
		#---
		#lqnote
		#back transformation 
		#gen_data = back_transformation(gen_data)
		#---
		geom   = gen_data[0,:,:]
		shiftE = gen_data[1,:,:]
		dipole = gen_data[2,:,:]
		geom = np.mat(geom)
		std = np.std(geom)
		print("std of geometry", std)
		basename = "gen_"+str(i).zfill(2)
		make_orca_input(gen_folder, basename, geom, atoms)
#		geomfile = "gen_xyz_"+i.zfill(2)+".xyz"
#		geomfile = os.path.join(gen_folder,geomfile)
#
#		inpfile = "gen_xyz_"+i.zfill(2)+".inp"
#		inpfile = os.path.join(gen_folder,inpfile)



		
	
