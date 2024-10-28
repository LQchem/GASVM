import numpy as np
import matplotlib.pyplot as plt
def calc_MAE(v1,v2):
    v1=np.array(v1)
    v2=np.array(v2)
    diff = v1-v2
    diff = np.abs(diff)
    sae  = np.sum(diff)
    mae  = sae/(len(v1))
    return mae
def calc_RMSE(v1,v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        a = v1 - v2
        rmse = np.sqrt(np.mean((a-np.mean(a))**2))
        return rmse


if __name__=="__main__":
	E1=np.loadtxt("E1.dat")
	E2=np.loadtxt("E2.dat")


	fig = plt.figure()
	dim = len(E1)
	x = np.arange(dim)
	plt.scatter(x, E1,label="E1")
	plt.scatter(x,E2,label="E2")
	plt.legend()
	plt.show()


	rmse = calc_RMSE(E1, E2)
	mae = calc_MAE(E1, E2)
	print("rmse", rmse)
	print("MAE", mae)

