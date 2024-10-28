import os
import torch
import pandas as pd
import json
from matplotlib.ticker import FormatStrFormatter as FMT

from F11_Generator_reformat import *
from F02_parameters import *
from F65_PES_NN import  PES_test2, do_normalization
def remove_short_bond_length(geom,thresh=1):
	monomer_idx1=[0,2,4,6,8]
	monomer_idx2=[1,3,5,7,9]
	for i in monomer_idx1:
		for j in monomer_idx1:
			if i==j:
				continue
			else:
				a1 = geom[i,:]	
				a2 = geom[j,:]
				d1 = a1 - a2	
				d1 = np.linalg.norm(d1)
				if d1 < thresh: 
					return 0
 
	for i in monomer_idx2:
		for j in monomer_idx2:
			if i==j:
				continue
			else:
				a1 = geom[i,:]	
				a2 = geom[j,:]
				d1 = a1 - a2	
				d1 = np.linalg.norm(d1)
				if d1 < thresh: 
					return 0
 
	return 1

def remove_long_OH(geom):
	pass	

def detect_outliers_zscore(data,case=2):
	if case==1:
		print("detect outlier by zscore")
		from scipy import stats
		z_scores = np.abs(stats.zscore(data))
		threshold = 3
		outliers = z_scores > threshold
	
		normal = z_scores <= threshold #normal is of bool list

	if case==2:
		print("detect outlier by percentile")
		ratio = 30 #should be less than 50
		q1 = np.percentile(data, ratio)  # 第一四分位数
		q3 = np.percentile(data, 100-ratio)  # 第三四分位数
		iqr = q3 - q1  # 四分位距
		lower_bound = q1 - 1.5 * iqr  # 下限
		upper_bound = q3 + 1.5 * iqr  # 上限
		outliers = (data < lower_bound) | (data > upper_bound)
		normal   = (data >= lower_bound) | (data <= upper_bound)
			

	outlier_indices = np.where(outliers)[0]
	normal_indices = np.where(normal)[0]

	return list(outlier_indices),list(normal_indices)

def plot_ED(y1,y2,ylabel='',title='tmp.png'):

	fig,ax=plt.subplots(figsize=(9,6))

	dim = len(y1)
	x= np.arange(dim)
	plt.scatter(x,y1,color='blue')
	plt.scatter(x,y2,color='orange')

	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.ylabel(ylabel,fontsize=20)
	plt.xlabel("points",fontsize=20)
	
	y_formatter = FMT('%1.3f')
	ax.yaxis.set_major_formatter(y_formatter)

	plt.savefig(title)
	plt.show()

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
	the input feature is of size 3N+1+3
	'''

	# 加载上次的参数作为默认值
	print("last param")
	last_params = load_params()
	print(last_params)	

	g_p, args = f02_main(json_flag=0)

	pth_file = "generator.pth"
	generator = torch.load(pth_file)
	print(generator)

	shift=373	
	
	gen_folder = "Gen_GASVM"
	os.makedirs(gen_folder,exist_ok=True)
	print("generated structrure saved to ", gen_folder)

	atoms=["H","H","H","H","O","O","O","O","C","C"]

	inp_dim = generator.out_Nfeature	

	#------
	#setup the back transformation function
	#g_p, args = f02_main(json_flag=0)
	if args.G_activation_func2 ==2:
		back_transformation = inv_sigmoid 
	elif args.G_activation_func2 ==1:
		back_transformation = inv_tanh
	#----

	#======add A16 ===========
	pth_file = "PES_NN.pth"
	NN_pes = torch.load(pth_file)
	count_paremeters(NN_pes)


	#=====add A16


	#``````````````````````
	count=0
	
	Ngen_config=100
	E_error_list = []
	D_error_list = []
	GAN_E_list = []
	PES_E_list = []
	GAN_D_list = []
	PES_D_list = []
	gen_data_list = []



	while(count<Ngen_config):	
		input_tensor = generator.generate_random(generator.batch_size, inp_dim) 
		#lq patch---
	#	input_tensor[:,:,-4] += -3
		#--lq patch

#		print("input_tensor shape",input_tensor.shape)
		#------lq patch---------------------------
		#add a constant to the init Energy to see if the energy can be shifted 
#		input_tensor[:,-4] = input_tensor[:,-4] + 0.3


		#-----------lq patch-----------------------


		batch_data = generator.forward(input_tensor).detach()
	#	print("batch_data shape",batch_data.shape)
	
		#```````````````````
	
#		print(generator.model)
		
		for i in range(generator.batch_size):
			gen_data = batch_data[i,:]
	#		print(gen_data)
			#---
			#lqnote
			#back transformation 
			#gen_data = back_transformation(gen_data)
			#---
			dipole = gen_data[-3:]
			GE     = gen_data[:-3]
			Energy = GE[-1]-shift
			flat_geom = GE[:-1]
			geom = flat_geom.view(-1,3)
			geom = np.mat(geom)
			std = np.std(geom)

			#===add A16 trail1===
	#		tmp  = flat_geom.unsqueeze(0) #add a dimension at  place 0
	##		bigX = torch.vstack([tmp,tmp])
	#		bigX = tmp
	#		pred_ED = NN_pes.forward(bigX)
	#	
	#		predE   = pred_ED[0,0]-shift
	#		predD   = pred_ED[0,1:]
			#===add A16===
#			if 0.96 < std < 1.5:
			if 1.1 < std <1.6:
				bond_length_flag = remove_short_bond_length(geom)
				if bond_length_flag == 1:
	
					basename = "gen_"+str(count).zfill(2)
					make_orca_input(gen_folder, basename, geom, atoms, Energy, dipole)
					gen_data_list.append(gen_data.detach().numpy())	
	
					#==== add A16 trail1====
		#			E_loss = predE-Energy 
		#			D_loss = predD-dipole
	
		#			D_norm_loss = torch.linalg.norm(predD) - torch.linalg.norm(dipole)
		#			
		#			PES_E_list.append(predE.detach().numpy()) 
		#			PES_D_list.append(predD.detach().numpy())
		#			GAN_E_list.append(Energy.detach().numpy())
		#			GAN_D_list.append(dipole.detach().numpy())
		#			E_error_list.append(E_loss.detach().numpy())
		#			D_error_list.append(D_norm_loss.detach().numpy())
					#=== add A16 ====
	
	
	#				print(count, " std of geometry", std, "E error ", E_loss.detach(), "D error", D_loss.detach())
					count+=1 
					if count == Ngen_config:
						break
	
	

	print("------------------------")	
	print("finish config generation")


	print("generate energy and dipole using GA-SVM geom and compare to PES-NN fit E and D")
#	print(len(gen_geom_list))
#	print(gen_geom_list[0].shape)

	remove_outlier=0
	if remove_outlier:
		# 检测异常值
		outlier_indices1,normal_indices1 = detect_outliers_zscore(PES_E_list)
		outlier_indices2,normal_indices2 = detect_outliers_zscore(GAN_E_list)
	
	#	union = set(normal_indices1) | set(normal_indices2) #get union
		union_outlier = set(outlier_indices1) | set(outlier_indices2)
		dim = len(PES_E_list)
		all_ind = set(np.arange(dim))
		union = all_ind - union_outlier
	
	
		trunc_PES =np.array( PES_E_list)
		trunc_GAN =np.array( GAN_E_list)
		union = list(union)
		print("len of normal indices",len(union))
		trunc_PES = trunc_PES[union]
		trunc_GAN = trunc_GAN[union]
	
	
		#plot the GA-svm data and the PES data
		plot_ED(trunc_PES,trunc_GAN,ylabel='Energy (a.u.)',title='Energy.png')
	
		
		a = trunc_PES - trunc_GAN
		rmse_test = np.sqrt(np.mean((a-np.mean(a))**2))
		print("rmse after removing outlier",rmse_test)
	


	count_paremeters(generator)

	print("compare GA-SVM results with PES NN results")
	gen_data_tensor = ConvertTensor(gen_data_list)
	norm_gen_data,norms = do_normalization(gen_data_tensor)
	e_loss, d_loss = PES_test2(norm_gen_data,NN_pes) 
	print("compare GA-SVM results with PES NN results after inverse normalization:")
	print(e_loss*torch.mean(norms), "rmse", torch.sqrt(e_loss*torch.mean(norms)))



