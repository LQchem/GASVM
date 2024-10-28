import copy
import torch 
from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import numpy as np
from F01_loaddata import * 

Ang2Bohr     = 1.8897161646320724
au2Kcal = 627.50
au2Debye     = 2.5412
class Feature(Dataset):

	def __init__(self,shift_value):
		self.load_data(shift_value)

	def load_data(self,shift_value):
		energy_tensor, shiftE_tensor,dipole_tensor, geom_list,Nconfig, atoms = load_data(shift_value)
		print("setup features")
		self.energy_tensor = energy_tensor
		self.shiftE_tensor = shiftE_tensor
		self.dipole_tensor = dipole_tensor
		self.geom_list     = geom_list  #members of geom_list are in tensor type of (N,3) shape

		self.Nconfig       = Nconfig
		self.atoms         = atoms
		self.NAtom         = len(atoms)
		self.frag1         = [0,2,4,6,8]
		self.frag2         = [1,3,5,7,9]
		self.mol1 = [ atoms[i] for i in self.frag1 ]
		self.mol2 = [ atoms[i] for i in self.frag2 ]
		
		
		#self.features could be type of list of type of tensor
		#if self.features is of type list, the member of the list should be of tensor type	
#		self.features = self.setup_feature2()
#		self.features = self.setup_feature1() #contains three parts: frag1 bond map upper triA (len=10), frag2 bond map upper triA(len=10), inter-mol dist(len=25)
#		self.feature_type(f_type=2)

	def select_feature_type(self,f_type=1):
		if f_type==1:
			#contains three parts: frag1 bond map upper triA (len=10), frag2 bond map upper triA(len=10), inter-mol dist(len=25)
			self.features = self.setup_feature1()

		if f_type==2:	
			#GED
			self.features = self.setup_feature2() #feature is of tensor type #energy is shifted
		 
		if f_type==3:
			# 2d geom, setup the GED_3channels as features
			self.features = self.setup_feature3()  #features is a list of tensor wish shape (3,Natom, 3) energy is shifted	


	def transform(self,data):

		#do the translation
		mean_vec = torch.mean(data,axis=0) #calc mean of each vector
		shift_data = data - mean_vec

		#do the exponential scaling
		shift_data = torch.exp(-shift_data)
		return shift_data	


	def setup_feature3(self):
		#setup the 3 channel features
		energy_tensor = self.energy_tensor 
		dipole_tensor = self.dipole_tensor
		geom_list     = self.geom_list
		shiftE_tensor = self.shiftE_tensor

		tensor_list = []
		shape = geom_list[0].shape
		ones  = torch.ones(*shape)
		for i in range(len(geom_list)):
			geom_channel   = geom_list[i]
			shiftE         = shiftE_tensor[i]
			dipole         = dipole_tensor[i,:]			
			shiftE_channel = shiftE * ones
			dipole_channel = dipole * ones
			GED_3channels  = torch.stack([geom_channel, shiftE_channel, dipole_channel])
			tensor_list.append(GED_3channels)

		return tensor_list			
	


	def setup_feature2(self,flag=0):
		#setup the GED as feature
		scalefac = 1
		energy_tensor = self.energy_tensor * scalefac
		shiftE_tensor = self.shiftE_tensor
		dipole_tensor = self.dipole_tensor * scalefac
		geom_list     = self.geom_list
		G_tensor      = torch.vstack(geom_list)
		G_tensor      = G_tensor.view(self.Nconfig,self.NAtom*3) * scalefac

		if flag==1:
			energy_tensor = self.transform(energy_tensor) 
			dipole_tensor = self.transform(dipole_tensor)
			G_tensor = self.transform(G_tensor)

			feature = torch.hstack([G_tensor, energy_tensor,dipole_tensor])

		else:
			feature = torch.hstack([G_tensor, shiftE_tensor,dipole_tensor])


		#do the normalization
		#feature = F.normalize(feature, p=2, dim=1) #p=2 uses L2 norm
		return feature 



	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	
	def setup_feature1(self):
		#use bond length as features.
		a1,a2,b1,b2,c1,c2,d1 = self.seperate_geometry(self.geom_list, self.frag1, self.frag2)

		self.mol1_geom_list = a1
		self.mol2_geom_list = a2

		self.bond_map_list1 = b1
		self.bond_map_list2 = b2

		self.upper_triA_list1 = c1 
		self.upper_triA_list2 = c2 

		self.inter_mol_dist_list = d1

		c1 = self.upper_triA_list1
		c2 = self.upper_triA_list2
		d1 = self.inter_mol_dist_list 

		feature_list = []
		for i,j,k in zip(c1,c2,d1):
			feature = torch.cat((i,j,k))
			feature_list.append(feature)
		return feature_list


	def calc_bond_map(self,geom): #b c
		NAtom,col = geom.shape
		bond_map  = torch.zeros(NAtom,NAtom)
		upper_triA=[]
		for i in range(NAtom):
			for j in range(i+1,NAtom):
				atom1 = geom[i,:]
				atom2 = geom[j,:]
				dist  = atom1 - atom2
				dist  = torch.linalg.norm(atom1-atom2)
				bond_map[i,j] = dist
				bond_map[j,i] = dist
				upper_triA.append(dist)
		return bond_map, torch.tensor(upper_triA)

	def calc_inter_mol_distance(self,geom1, geom2): #d1
	
		NAtom = geom1.shape[0]
		dist_list = []
		for i in range(NAtom):
			atomi = geom1[i,:]
			for j in range(NAtom):
				atomj = geom2[j,:]
				dist = atomi-atomj
				dist = torch.linalg.norm(dist)
				dist_list.append(dist)

		return torch.tensor(dist_list)

	def seperate_geometry(self,geom_list,frag1,frag2):
		a1 = geom1_list = []
		a2 = geom2_list = []

		b1  = bond_map_list1=[]
		b2  = bond_map_list2=[]

		c1  = upper_triA_list1 = []
		c2  = upper_triA_list2 = []

		d1 = inter_mol_dist_list = []

		for geom in geom_list:
			geom1 = geom[frag1,:]
			geom2 = geom[frag2,:]

			geom1_list.append(geom1) #a1
			geom2_list.append(geom2) #a2

			#-----
			#calc bond map
			bond_map1, upper_triA1 = self.calc_bond_map(geom1) #mol1
			bond_map2, upper_triA2 = self.calc_bond_map(geom2) #mol2

			bond_map_list1.append(bond_map1)  #mol1 b1 #bond_map is of type tensor
			bond_map_list2.append(bond_map2)  #mol2 b2

			upper_triA_list1.append(upper_triA1) #mol1 c1 #upper_triA is of type tensor
			upper_triA_list2.append(upper_triA2) #mol2 c2


			inter_mol_dist = self.calc_inter_mol_distance(geom1, geom2) #inter molecular distance
			inter_mol_dist_list.append(inter_mol_dist) #d1 #inter_mol_dist if os type tensor
			#----

#		e1 = copy.deepcopy(upper_triA_list1) # 
#		e1.extend(upper_triA_list2) #e1 = cat(c1,c2)

		return a1,a2,b1,b2,c1,c2,d1#,e1 
	#end of setup_feature1
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#---------
	def calc_Gaussian_model(self,data):
		self.mean_vec = torch.mean(data,axis=0) #calc mean of each vector
		self.cov_mat  = torch.cov(data.T) #data is of shape (Nsampe, Nfeature), so need to do the transformation

	#   if calc mean and cov for the whole datalist:
	#	alldata = torch.vstack(my_datalist)
	#	mu2,sigm2 = feature.calc_Gaussian_model(alldata)

		return self.mean_vec, self.cov_mat

	#calc probability density function
	#mean_vec: the mean vector
	#cov_mat: the covariance matrix
	def calc_PDF(self,data, mean_vec, cov_mat):
		
#		cov_mat = cov_mat + 1e-8*torch.eye(cov_mat.size(0)) #add a small number on the diagonal terms 
		mvn = MultivariateNormal(loc=mean_vec, covariance_matrix = cov_mat)
		pdf_values = mvn.log_prob(data).exp()
		return pdf_values


	def positive_definite_mat_check(self,mat):
		from torch.linalg import eigvals
		eigenvalues = eigvals(mat)
		print(eigenvalues)
		if torch.all(eigenvalues>0):
			pass
		else:
			print("matrix is not positive denfinite")


	#---------


if __name__=="__main__":

	shift_value=379
	feature = Feature(shift_value)

	feature.select_feature_type(3)
	print(feature.features[0].shape)
	print(feature.features[0][0,:,:])
	print("--")
	print(feature.features[0][1,:,:])
	print("--")
	print(feature.features[0][2,:,:])

	print("fype ==2")
	feature.select_feature_type(2)
	print(feature.features[0].shape)
	print(feature.features[0])




#	my_datalist = feature.features
#	mu,sigma = feature.calc_Gaussian_model(my_datalist)
#	
#
#	batch_size=64
#	
#	dataloader_l = setup_dataloader(my_datalist,batch_size,shuffle=False)

#	print("pdf values")
#	for i,data in enumerate(dataloader_l):
#		pdf_values = feature.calc_PDF(data,mu,sigma)
#		print(pdf_values)	
#		if i==1:
#			break
