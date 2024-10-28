import numpy as np
#from sympy import symbols, Eq, solve, simplify
from scipy.optimize import fsolve,root
def printmat(mat):
        r,c=mat.shape
        for i in range(r):
                for j in range(c):
                        print("%10.6f" %mat[i,j],"   ",end='')
                print()

def make_equations(xyz1,xyz2,xyz3,r1,r2,r3):
	x1,y1,z1 = xyz1
	x2,y2,z2 = xyz2
	x3,y3,z3 = xyz3

	def equations(vars):
		x,y,z = vars
		eq1 =  (x-x1)**2+(y-y1)**2+(z-z1)**2 - r1**2 
		eq2 =  (x-x2)**2+(y-y2)**2+(z-z2)**2 - r2**2 
		eq3 =  (x-x3)**2+(y-y3)**2+(z-z3)**2 - r3**2
		return [eq1,eq2,eq3]

	return equations 

def make_equations2(xyz1,xyz2,r1,r2):
	x1,y1,z1 = xyz1
	x2,y2,z2 = xyz2


	def equations(vars):
		x,y = vars
		eq1 =  (x-x1)**2+(y-y1)**2 - r1**2 
		eq2 =  (x-x2)**2+(y-y2)**2 - r2**2 

		return [eq1,eq2]

	return equations 


def assign_first4atoms(bond_map):
	NAtom = bond_map.shape[0]
	geom = np.zeros((NAtom,3)) #0

	geom[1,0] = bond_map[1,0] #1
	init_guess = [bond_map[1,0],bond_map[2,0]] #2
	equation_w_para = make_equations2(geom[0,:], geom[1,:],bond_map[2,0],bond_map[2,1])
	solutions = fsolve(equation_w_para,init_guess)
	geom[2,0]=solutions[0]
	geom[2,1]=solutions[1]

	init_guess = [bond_map[3,0],bond_map[3,1],bond_map[3,2]] #3
	equation_w_para = make_equations(geom[0,:],geom[1,:],geom[2,:],bond_map[3,0],bond_map[3,1],bond_map[3,2])
	solutions = fsolve(equation_w_para,init_guess)
	geom[3,:]=solutions

	return geom

#using X-A, X-B, X-C distances to solve xyz for X would give two equi. solutions, which is symmetric to the ABC plane.
#this funciton is to remove one solution by comparing the distance the the forth atom, namely X-D distance
#on input:
# solution, the coordinate of X
# Dcoord, the coordinate of D atom
# distXD, the distance of X-D
def determine_unique_solution(solution,Dcoord, distXD):
	x,y,z = solution
	
	dist1 = np.array([x,y,z])-Dcoord
	dist1 = np.linalg.norm(dist1)

	dist2 = np.array([x,y,-z])-Dcoord
	dist2 = np.linalg.norm(dist2)

	dist3 = np.array([x,-y,z])-Dcoord
	dist3 = np.linalg.norm(dist3)

	dist4 = np.array([-x,y,z])-Dcoord
	dist4 = np.linalg.norm(dist4)

	flag=1

	if abs(dist1-distXD) < 0.001:
		return np.array([x,y,z])
	elif abs(dist2-distXD) < 0.001:
		return np.array([x,y,-z])
	elif abs(dist3-distXD) < 0.001:
		return np.array([x,-y,z])
	elif abs(dist4-distXD) < 0.001:
		return np.array([-x,y,z])

	else:
		flag=0

	assert flag, "the determination of unique solution has problem"



def assign_remaining_atoms(geom,bond_map):
	NAtom = bond_map.shape[0]

	j=0  #determine the set of reference atoms
	for i in range(4,NAtom):
		init_guess = [bond_map[3,0],bond_map[3,1],bond_map[3,2]]
		equation_w_para = make_equations(geom[0,:],geom[1,:],geom[2,:],bond_map[i,0],bond_map[i,1],bond_map[i,2])
		solutions = fsolve(equation_w_para,init_guess)
		
		unique_sol = determine_unique_solution(solutions,geom[3,:],bond_map[i,3])

		geom[i,:]=unique_sol
	return geom
	


if __name__=="__main__":
	'''
	read in a bond length map and return the xyz coordinate
	the bond length map is an N*N symmetric matrix, where each entry denotes rij
	/home/lq/project/conformation-gan/sqr-map-input/metal-clusters/F02_graph_to_Cartesian.py , I rename the file
	'''

	filename="conform_002.dat"
	bond_map = np.loadtxt(filename)

	geom = assign_first4atoms(bond_map)		
	assign_remaining_atoms(geom,bond_map)
	
	atoms="Al"
	outfile = "test002.xyz"
	fw=open(outfile,"w")
	NAtom = geom.shape[0]
	fw.write(str(NAtom))
	fw.write("\n\n")
	for i in range(NAtom):
		line=atoms+f"  {geom[i,0]:>10f}  {geom[i,1]:>10f}  {geom[i,2]:>10f}\n"
		fw.write(line)





