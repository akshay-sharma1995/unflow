import torch
import torch.nn.functional as F
import numpy as np
import time
import skimage.io
from losses import *
from matplotlib import pyplot as plt
import cv2
from image_warp import *


import torch
import os
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import flow_transforms
import skimage.io
import skimage.color    

import numpy as np
import random
from matplotlib import pyplot as plt
from FlownetC import *
import torch.optim as optim
import torch
from train import *
from util import rgb_to_y, visualize_op_flow


def evaluate(train_loader, model):

	# switch to train mode
	train_loader = torch.tensor(train_loader).type(torch.cuda.FloatTensor)
	## train_loader.size = Bx3x(C==3)xHxW
	model.eval()

	# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	# epoch = checkpoint['epoch']
	# loss = checkpoint['loss']

	out_flow12 = model(train_loader[:,0:2]) ## list of 5 optical flows not of image size
	out_flow23 = model(train_loader[:,1:3]) ## list of 5 optical flows not of image size


	# print("out_flow12.shape",out_flow12.size())
	h, w = train_loader.size()[-2:] ## image size

	print("h,w",h,w)
	
	out_flow12 = F.interpolate(out_flow12, (h,w)) ## upsampling flow to image size
	out_flow23 = F.interpolate(out_flow23, (h,w)) ## upsampling flow to image size

	out_image = []
	im1 = train_loader[:,0] ## the 1st image of the triplets in the batch
	im2 = train_loader[:,1] ## the 2nd image of the triplets in the batch
	im3 = train_loader[:,2] ## the 3rd image of the triplets in the batch

	# weights = [0.32, 0.08, 0.02, 0.01, 0.005]

	loss = 0
	combined_loss = []


	warped_img12 = image_warp(im1,out_flow12)
	warped_img23 = image_warp(im2,out_flow23)


	criterion = torch.nn.MSELoss()

	loss12 = criterion(im2,warped_img12)
	loss23 = criterion(im3,warped_img23)



	return loss12, loss23, warped_img12, warped_img23, out_flow12, out_flow23  	


def imlist(fpath):
	flist = os.listdir(fpath)
	return flist

def main():
	model = FlowNetC()
	# s_loss = 
	if torch.cuda.is_available():
		model.cuda()
		print("Model shifted to GPU")

	# optimizer = optim.Adam(model.parameters(),lr=1e-4)
	PATH = "./checkpoints_no_temp_loss/t_loss_model10.pth"
	DIR = "../../../Project_data/evaluation_data/npz_3_set/"
	checkpoint = torch.load(PATH)
	model.load_state_dict(checkpoint['model_state_dict'])

	batch_size = 1
	# length =
	flist = imlist(DIR)
	# print("flist",flist)
	number_of_image_sets = len(flist)
	idxs = (np.arange(1,number_of_image_sets,1)) ## id of all image_sets
	# random.shuffle(idxs) ## shuffling the idxs
	# num_epochs = 10
	# batches_processed = 0
	epoch_loss_list = []
	for epoch in range(0,1):
		# epoch_loss = 0
		for i in range(len(idxs)):
			image_batch = []
			if(len(idxs)-i>=batch_size):
				count = batch_size
			else:
				count = len(idxs) - i
			for j in range(count): ## making batches
				path = DIR + flist[idxs[i]]
				image_triplet = np.load(path)['arr_0']
				image_triplet[0] = skimage.color.rgb2ycbcr(image_triplet[0])
				image_triplet[1] = skimage.color.rgb2ycbcr(image_triplet[1])
				image_triplet[2] = skimage.color.rgb2ycbcr(image_triplet[2])

				image_triplet = image_triplet[:,:,:,0:1] ## only y channel of each image

				## image_triplet.size = 3xHxWx3
				## mapping the images to (-1,1)
				# image_triplet = ((image_triplet.astype(np.float64) - 16.0) / (235.0-16.0))   * 2.0 - 1.0
				# print("max_min", np.nanmax(image_triplet),np.nanmin(image_triplet))
				image_batch.append(image_triplet)

			image_batch = np.array(image_batch)
			# print("image_batch.shape",np.shape(image_batch))



			warped_img_direc = "./warped_images_no_t_loss/"+str(i)+"/"
			if not(os.path.exists(warped_img_direc)):
				os.mkdir(warped_img_direc)

			skimage.io.imsave(warped_img_direc+"original2"+".png",image_batch[0,1,:,:,0])
			skimage.io.imsave(warped_img_direc+"original3"+".png",image_batch[0,2,:,:,0])

			image_batch = ((image_batch.astype(np.float64) - 16.0) / (235.0-16.0))   * 2.0 - 1.0
			image_batch = np.rollaxis(image_batch,4,2)


			# print("listofimages.shape",np.shape(image_batch))
			# batches_processed += 1
			# print("image_batch.shape",np.shape(image_batch))
			loss12, loss23, warped_img12, warped_img23, out_flow12, out_flow23 = evaluate(image_batch,model)

			out_flow12 = out_flow12.permute(0,2,3,1)[0]  ## batch dimension removed. size--> h,w,2
			out_flow23 = out_flow23.permute(0,2,3,1)[0]

			vis_out_flow12 = visualize_op_flow(out_flow12.detach().cpu().numpy())
			vis_out_flow23 = visualize_op_flow(out_flow23.detach().cpu().numpy())

			cv2.imwrite(warped_img_direc+"op_flow12"+".png",vis_out_flow12)
			cv2.imwrite(warped_img_direc+"op_flow23"+".png",vis_out_flow23)




			print("loss12",loss12)
			print("loss23",loss23)

			# print("out_flow12",out_flow12.size())
			
			# print("warped_images.shape",warped_img12.size())
			warped_img12 = 0.5*(warped_img12 + 1)*(235.0-16.0) + 16.0
			warped_img23 = 0.5*(warped_img23 + 1)*(235.0-16.0) + 16.0
	
			warped_img12 = warped_img12.type(torch.cuda.IntTensor).permute(0,2,3,1)
			warped_img23 = warped_img23.type(torch.cuda.IntTensor).permute(0,2,3,1)


			skimage.io.imsave(warped_img_direc+"warped12"+".png",warped_img12[0,:,:,0].detach().cpu().numpy())
			skimage.io.imsave(warped_img_direc+"warped23"+".png",warped_img23[0,:,:,0].detach().cpu().numpy())




			# epoch_loss += batch_loss
			

if __name__ == '__main__':
	main()
