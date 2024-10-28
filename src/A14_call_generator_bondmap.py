import os
import torch
import pandas as pd
from F11_Generator_reformat import *
from F02_parameters import *
from F03_feature import *
from F04_graph_to_Cartesian import *
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

def symmetrize_triA(NAtom,triAngle):
	bond_map = np.mat(np.zeros((NAtom,NAtom)))
	count=0
	for i in range(NAtom):
		for j in range(i+1,NAtom):
			bond_map[i,j] = triAngle[count]	
			bond_map[j,i] = triAngle[count]
			count+=1
	return bond_map

def bond_map2xyz(bond_map):
	geom = assign_first4atoms(bond_map)
	assign_remaining_atoms(geom,bond_map)
	return geom



if __name__=="__main__":
	'''
	uses F11 generator
	the input feature is of size triA+triA+inter_mol_dis = 10+10+25
	'''

	pth_file = "generator.pth"
	generator = torch.load(pth_file)


	shift=379	
	
	gen_folder = "Gen_GASVM"
	os.makedirs(gen_folder,exist_ok=True)
	print("generated structrure saved to ", gen_folder)

	atoms=["H","H","H","H","O","O","O","O","C","C"]
	monomer=["H","H","O","O","C"]
	batch_size=10

	inp_dim = generator.latent_dim	
	input_tensor = generator.generate_random(batch_size, inp_dim)


	if generator.model_flag==203:
		input_tensor = input_tensor.unsqueeze(1) #add a dimension at  place 1, thus gen_data shape is (batch_size, 1, out_feature)

	batch_data = generator.forward(input_tensor).detach()
	
	if generator.model_flag==203:
		batch_data = batch_data.squeeze(1) #add a dimension at  place 1, thus gen_data shape is (batch_size, 1, out_feature)



	print(generator.model)
	dim1=10 #triA len
	dim2=10 #triA len
	dim3=25 #inter mol dist len

	#-------------
#	shift_value=379
#	feature = Feature(shift_value)
#	triU = feature.mol1_geom_list[0].detach().numpy()
#	bond_map = symmetrize_triA(len(monomer), np.ravel(triU))
#	geom     = bond_map2xyz(bond_map)
#	printmat(geom)
	#-------------
	for i in range(batch_size):
		gen_data = batch_data[i,:]
		frag1 = gen_data[0:dim1]
		frag2 = gen_data[dim1:dim2]
		inter_mol_dist= gen_data[dim2:]

		
		bond_map = symmetrize_triA(len(monomer), frag1)
		print(bond_map)
		geom     = bond_map2xyz(bond_map)
		printmat(geom)
		break

	#	geom = np.mat(geom)
	#	std = np.std(geom)
	#	print("std of geometry", std)
	#	#basename = "gen_"+str(i).zfill(2)
	#	#make_orca_input(gen_folder, basename, geom, atoms)
