import os
import torch
import pandas as pd
#from F10_Generator import *
from F11_Generator_reformat import *
from torchvision.utils import save_image

if __name__=="__main__":

	pth_file = "mnist.pth"
	generator = torch.load(pth_file)
	print(generator.model)
	inp_dim = generator.out_Nfeature 
	batch_size=generator.batch_size
	
	gen_folder = "Gen_mnist"
	os.makedirs(gen_folder,exist_ok=True)
	print("generated MNIST data saved to ", gen_folder)

	if generator.model_flag==204:
		inp_dim = 28

	gen_data = generator.generate_random(batch_size, inp_dim)
	print(gen_data.shape)

#	if generator.model_flag==203:
#		gen_data = gen_data.unsqueeze(1) #add a dimension at  place 1, thus gen_data shape is (batch_size, 1, out_feature)

	gen_data = generator.forward(gen_data)

#	if generator.model_flag==203:
#		gen_data = gen_data.squeeze(1) #add a dimension at  place 1, thus gen_data shape is (batch_size, 1, out_feature)



	print(gen_data.shape)	
	for i in range(batch_size):
		img = gen_data[i,:]
		img = img.view(28,28)
		filename=f"test_{i}.png"
		filename = os.path.join(gen_folder, filename)
		save_image(img,filename,normalize=True)

#	gen_data = gen_data.view(batch_size,28,28)
#	
#	print(gen_data.shape)
#	save_image(gen_data,filename)

