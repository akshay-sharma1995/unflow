import torch
import torch.nn.functional as F
import numpy as np
import time
import skimage.io
from losses import *
from matplotlib import pyplot as plt
import cv2
from image_warp import *
# def display_flow(u,v,x_orig):
# 	hsv = np.zeros(x_orig.size(), dtype=np.uint8)
# 	hsv = np.rollaxis(np.rollaxis(hsv,1,0),2,1)
# 	print("hsv",hsv.shape)
# 	hsv[..., 1] = 255
# 	print("hsv",hsv.shape)
# 	print("u.shape",u.shape)
# 	mag, ang = cv2.cartToPolar(u, v)
# 	print("mag.shape",np.shape(mag),np.shape(ang))
# 	hsv[..., 0] = ang * 180 / np.pi / 2
# 	hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
# 	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# 	cv2.imshow("colored flow", bgr)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

def train(train_loader, model, optimizer):

	# switch to train mode
	train_loader = torch.tensor(train_loader).type(torch.cuda.FloatTensor)
	## train_loader.size = Bx3x(C==3)xHxW
	model.train()


	out_flow12 = model(train_loader[:,0:2]) ## list of 5 optical flows not of image size
	out_flow23 = model(train_loader[:,1:3]) ## list of 5 optical flows not of image size

	h, w = train_loader.size()[-2:] ## image size
	
	out_flow12 = [F.interpolate(oflow, (h,w)) for oflow in out_flow12] ## upsampling flow to image size
	out_flow23 = [F.interpolate(oflow, (h,w)) for oflow in out_flow23] ## upsampling flow to image size

	out_image = []
	im1 = train_loader[:,0] ## the 1st image of the triplets in the batch
	im2 = train_loader[:,1] ## the 2nd image of the triplets in the batch
	im3 = train_loader[:,2] ## the 3rd image of the triplets in the batch

	weights = [0.32, 0.08, 0.02, 0.01, 0.005]

	combined_loss = []
	for i in range(0,len(out_flow12)):
		oflow12 = out_flow12[i]
		oflow23 = out_flow23[i]

		warped_img12 = image_warp(im1,oflow12)
		warped_img23 = image_warp(im2,oflow23)

		loss = flow_loss(train_loader,warped_img12,warped_img23,oflow12,oflow23,weights[i])  
		combined_loss.append(loss)

	print("combined_loss",combined_loss)

	total_loss = torch.sum(combined_loss)
	optimizer.zero_grad()
	total_loss.backward()
	optimizer.step()		  	
	# warped_flow = image_warp(oflow,oflow)
	# warped_im32 = image_warp(im3,warped_flow)
	# warped_img2 = image_warp(orig_img,oflow)
	# warped_img3 = image_warp(warped_img2,warped_flow)
	# warped_img = (warped_img + 1) / 2
	# warped_img_ = 255*((warped_img[:]) + 1) / 2
	# warped_img_ = warped_img_.type(torch.cuda.IntTensor)

	# warped_img3_ = 255*((warped_img3[:]) + 1) / 2
	# warped_img3_ = warped_img3_.type(torch.cuda.IntTensor)

	# for p in range(0,5):
	#     ofnp = out_flow[p].permute(0,2,3,1)
	#     flow_h = ofnp.size()[1]
	#     print('flow_h',flow_h)
	#     flow_w = ofnp.size()[2]
	#     ofnp[:,0:1] = (ofnp[:,0:1] - torch.min(ofnp[:,0:1])[0])
	#     ofnp[:,0:1] = (2*ofnp[:,0:1] / torch.max(ofnp[:,0:1])[0]) - 1

	#     # if(ofnp[:,0:1]>1):
	#     #      ofnp[:,0:1]=1
		
	#     ofnp[:,1:2] = (ofnp[:,1:2] - torch.min(ofnp[:,1:2])[0])
	#     ofnp[:,1:2] = (2*ofnp[:,1:2] / torch.max(ofnp[:,1:2])[0]) - 1

	#     # ofnp[:,1:2] = ofnp[:,1:2]/flow_w 
	#     # ofnp = np.meshgrid(np.linspace(0,0,flow_w), np.linspace(-0,0,flow_h))
	#     # ofnp = torch.tensor(ofnp).type(torch.cuda.FloatTensor)
	#     # ofnp = ofnp.unsqueeze(0)
	#     # ofnp = ofnp.permute(0,2,3,1)
	#     # print('ofnp.sahpe',ofnp.size())

	#     # ofnp = torch.clamp(ofnp,min=-0.0,max=0.0)
	#     # ofnp = (ofnp/torch.max(ofnp)[0])*2-1
	#     # ofnp =  
	#     print('Max',torch.max(ofnp))
	#     print('Min',torch.min(ofnp))
	#     out_image.append(F.grid_sample((train_loader[:,0]+1)/2,ofnp))

 

	# x_orig = train_loader[0,1]
	# x_orig = 255.0*((x_orig) + 1) / 2
	# x_orig = x_orig.type(torch.cuda.IntTensor)
	# skimage.io.imsave("./orginial_img.png",np.rollaxis(np.rollaxis(x_orig.cpu().data.numpy(),1,0),2,1))
	# # skimage.io.imshow(warped_img[0].permute(1,2,0).cpu().data.numpy())
	# # plt.show()

	# # xi = out_image[0][0,:]
	# # xi = 255.0*(xi + 1.0)/2.0    
	# # xi = xi.type(torch.cuda.IntTensor)
	# skimage.io.imsave("./warped_img.png",warped_img_[0].permute(1,2,0).cpu().data.numpy())
	# skimage.io.imsave("./warped_img3.png",warped_img3_[0].permute(1,2,0).cpu().data.numpy())
	
	# plt.show()
	# plt.show()
	# u = out_flow[0][0,0].cpu().data.numpy()
	# v = out_flow[0][0,1].cpu().data.numpy()

	# u = np.rollaxis(np.rollaxis(out_flow[0][0,0].cpu().data.numpy(),1,0),2,1)
	# v = np.rollaxis(np.rollaxis(out_flow[0][0,1].cpu().data.numpy(),1,0),2,1)
	# display_flow(u,v,x_orig)
	# hsv = np.zeros(x_orig.size(), dtype=np.uint8)

	# hsv = np.rollaxis(np.rollaxis(hsv,1,0),2,1)
	# print("hsv",hsv.shape)

	# hsv[..., 1] = 255
	# print("hsv",hsv.shape)
	# print("u.shape",u.shape)
	# mag, ang = cv2.cartToPolar(u, v)
	# print("mag.shape",np.shape(mag),np.shape(ang))
	# hsv[..., 0] = ang * 180 / np.pi / 2
	# hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	# bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	# cv2.imshow("colored flow", bgr)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()





'''    for i, input in enumerate(train_loader):
		# measure data loading time
		# data_time.update(time.time() - end)

		# input = input.to(device)
		
		print("input",input.shape)
		# end
		# compute output
		out_flow = model(input)
		# if args.sparse:
			# Since Target pooling is not very precise when sparse,
			# take the highest resolution prediction and upsample it instead of downsampling target
		h, w = input.size()[-2:]
		out_flow = [F.interpolate(out_flow[0], (h,w)), *out_flow[1:]]
		# out_image = []
		# out_image.append(F.grid_sample(input,out_flow[0]))
		# out_image.append(F.grid_sample(input,out_flow[1]))
		# out_image.append(F.grid_sample(input,out_flow[2]))
		# out_image.append(F.grid_sample(input,out_flow[3]))
		# out_image.append(F.grid_sample(input,out_flow[4]))
		# out_image.append(F.grid_sample(input,out_flow[5]))

		# loss_image = loss(target,out_image)
		# loss = multiscaleEPE(output, target, weights=args.multiscale_weights, sparse=args.sparse)
		# flow2_EPE = args.div_flow * realEPE(output[0], target, sparse=args.sparse)
		# # record loss and EPE
		# losses.update(loss.item(), target.size(0))
		# train_writer.add_scalar('train_loss', loss.item(), n_iter)
		# flow2_EPEs.update(flow2_EPE.item(), target.size(0))

		# compute gradient and do optimization step
		optimizer.zero_grad()
		# loss_image.backward()
		optimizer.step()

		# measure elapsed time
		# batch_time.update(time.time() - end)
		# end = time.time()

		# if i % args.print_freq == 0:
		#     print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}'
		#           .format(epoch, i, epoch_size, batch_time,
		#                   data_time, losses, flow2_EPEs))
		# n_iter += 1
		# # if i >= epoch_size:
		#     break
	loss_image = 0
	return loss_image
'''