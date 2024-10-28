import os
import torch
import pandas as pd
import json
from F11_Generator_reformat import *
from F02_parameters import *
from F03_feature import *

def count_paremeters(model):
	num = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("total number of model parameters ", num) 


def load_params():
    """从文件加载参数"""
    if os.path.exists('params.json'):
        with open('params.json', 'r') as f:
            return json.load(f)
    return {}

def inv_sigmoid(t):
	return torch.log( (1/t)-1 )
def inv_tanh(t):
	return torch.atanh(t)

def make_orca_input(gen_folder,basename,geom,atoms, Energy, dipole):
	
	xyzfile = basename+".xyz"
	inpfile = basename+".inp"
	xyzfile = os.path.join(gen_folder, xyzfile)
	inpfile = os.path.join(gen_folder, inpfile)

	eng = str(Energy)
	dip = str(dipole)

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
	fw.write(f"#predicted energy {eng}, predicted dipole {dip}\n")
	fw.write("!pal DLPNO-CCSD(T) aug-cc-pvtz aug-cc-pvtz/c rijk aug-cc-pvtz/jk tightpno TightSCF nopop\n")
	fw.write("!angs\n")
	fw.write("%pal nproc 4 end\n")
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
	'''

	# 加载上次的参数作为默认值
#	print("last param")
#	last_params = load_params()
#	print(last_params)	

	g_p, args = f02_main(json_flag=0)

	pth_file = "PES_NN.pth"
	generator = torch.load(pth_file)


	shift=379	
	

	
	atoms=["H","H","H","H","O","O","O","O","C","C"]

	inp_dim = generator.out_Nfeature	

	print("load my data")
	shift_value=379

	feature = Feature(shift_value)

	#use Feature class and use GED as features; energy  is shifted
	print("use GED as features")
	feature_type=2
	feature.select_feature_type(f_type=feature_type)
	my_datalist = feature.features 
	out_feature = len(my_datalist[0])   #len = 3N+1+3

	batch_size = g_p.batch_size
	dataloader_l = setup_dataloader(my_datalist,batch_size) #for linear NN
	

	for idx,batch_tensor in enumerate(dataloader_l):  #batch_tensor is of 3N+1+3 if linear NN is used

		bigY = batch_tensor[:,-4:]
		bigX = batch_tensor[:,0:-4]

		out = generator.forward(bigX)
		g_loss = nn.MSELoss()(out,bigY) 

		print(
	                " [G loss: %f]  "
	                 % (  g_loss.item()))
	




