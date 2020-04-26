import torch
import torch.nn.functional as F
import numpy as np
import time
import skimage.io
from losses import *
from matplotlib import pyplot as plt
from image_warp import *
import os

from util import save_samples
epoch_num = 0
batch_num = 0
def train(train_loader, model, optimizer):
	global batch_num, epoch_num
	batch_num = (batch_num+1) % 149
	if(batch_num == 0):
		epoch_num += 1
	# switch to train mode
	# train_loader = torch.tensor(train_loader).type(torch.cuda.FloatTensor)
	## train_loader.size = Bx3x(C==3)xHxW
	model.train()
	# print(train_loader.shape)
	shape = train_loader.shape
	train_loader = torch.reshape(train_loader, (shape[0], shape[1], 1, shape[2], shape[3]))
	out_flow12 = model(train_loader)
	# out_flow12 = model(train_loader[:,0:2]) ## list of 5 optical flows not of image size
	# out_flow23 = model(train_loader[:,1:3]) ## list of 5 optical flows not of image size

	h, w = train_loader.size()[-2:] ## image size
	
	out_flow12 = [F.interpolate(oflow, (h,w)) for oflow in out_flow12] ## upsampling flow to image size
	# out_flow23 = [F.interpolate(oflow, (h,w)) for oflow in out_flow23] ## upsampling flow to image size

	out_image = []
	im1 = train_loader[:,0] ## the 1st image of the triplets in the batch
	im2 = train_loader[:,1] ## the 2nd image of the triplets in the batch
	# im3 = train_loader[:,2] ## the 3rd image of the triplets in the batch

	weights = [0.32, 0.08, 0.02, 0.01, 0.005]
	folder_path = "/home/suhail/DL/unflow/data"
	if(not os.path.exists(folder_path)):
		os.mkdir(folder_path)
	loss = 0
	combined_loss = []
	for i in range(0,len(out_flow12)):
		oflow12 = out_flow12[i]
		# oflow23 = out_flow23[i]

		warped_img12 = image_warp(im1,oflow12)
		# warped_img23 = image_warp(im2,oflow23)

		temp_loss = flow_loss_2_frames(im1, im2, warped_img12, oflow12)
		# temp_loss = flow_loss(train_loader,warped_img12,warped_img23,oflow12,oflow23,weights[i])  
		loss += temp_loss
		combined_loss.append(temp_loss/weights[i])

	# print("combined_loss",combined_loss)

	# total_loss = torch.sum(torch.tensor(combined_loss).type(torch.cuda.FloatTensor))
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()	
	# print(loss.grad)
	# print(loss)
	for i in range(oflow12.shape[0]):
		# print(warped_img12[0][0].shape)
		save_samples(warped_img12[0][i].clone().detach().numpy(), folder_path, epoch_num, batch_num, "predicted")
		save_samples(im1[0][i].clone().detach().numpy(), folder_path, epoch_num, batch_num, "actual_frame1")
		save_samples(im2[0][i].clone().detach().numpy(), folder_path, epoch_num, batch_num, "actual_frame2")

	return loss.item()	  	
