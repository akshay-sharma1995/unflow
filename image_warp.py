import numpy
import torch

def image_warp(im,flow):
	"""Performs a backward warp of an image using the predicted flow.

	Args:
		im: Batch of images. [num_batch, channels, height, width]
		flow: Batch of flow vectors. [num_batch, 2, height, width]
	Returns:
		warped: transformed image of the same shape as the input image.
	"""

	## may think of swaping image axes
	cuda0 = torch.device('cuda:0')
	im = im.permute(0,2,3,1) ## [num_batch, height, width, channels]
	flow = flow.permute(0,2,3,1)
	num_batch, height, width, channels = im.size()
	max_x = int(width - 1)
	max_y = int(height - 1)
	zero = torch.zeros([],dtype=torch.int32)
	zero = zero.type(torch.cuda.IntTensor)

	im_flat = torch.reshape(im,(-1,channels))
	flow_flat = torch.reshape(flow,(-1,2))

	flow_floor = torch.floor(flow_flat).type(torch.cuda.IntTensor)
	bilinear_weights = flow_flat - torch.floor(flow_flat)

	pos_x = torch.arange(width).repeat(height*num_batch).type(torch.cuda.IntTensor)
	grid_y = torch.arange(height).unsqueeze(1).repeat(1,width).type(torch.cuda.IntTensor)
	pos_y = torch.reshape(grid_y,(-1,)).repeat(num_batch).type(torch.cuda.IntTensor)

	x = flow_floor[:,0]
	y = flow_floor[:,1]
	xw = bilinear_weights[:,0]
	yw = bilinear_weights[:,1]

	wa = ((1-xw)*(1-yw)).unsqueeze(1)
	wb = ((1-xw)*yw).unsqueeze(1)
	wc = (xw*(1-yw)).unsqueeze(1)
	wd = (xw*yw).unsqueeze(1)

	x0 = pos_x + x
	x1 = x0 + 1
	y0 = pos_y + y
	y1 = y0 + 1

	x0 = torch.clamp(x0, zero, max_x)
	x1 = torch.clamp(x1, zero, max_x)
	y0 = torch.clamp(y0, zero, max_y)
	y1 = torch.clamp(y1, zero, max_y)

	dim1 = width * height
	batch_offsets = torch.arange(num_batch) * dim1
	base_grid = batch_offsets.unsqueeze(1).repeat(1,dim1)
	base = torch.reshape(base_grid, (-1,)).type(torch.cuda.IntTensor)

	base_y0 = base + y0 * width
	base_y1 = base + y1 * width
	idx_a = base_y0 + x0
	idx_a = idx_a.unsqueeze(1).repeat(1,channels)
	idx_b = base_y1 + x0
	idx_b = idx_b.unsqueeze(1).repeat(1,channels)
	idx_c = base_y0 + x1
	idx_c = idx_c.unsqueeze(1).repeat(1,channels)
	idx_d = base_y1 + x1
	idx_d = idx_d.unsqueeze(1).repeat(1,channels)

	print(im_flat.size())
	print(idx_a.size())

	Ia = torch.gather(im_flat, dim=0, index=idx_a.type(torch.cuda.LongTensor))
	Ib = torch.gather(im_flat, dim=0, index=idx_b.type(torch.cuda.LongTensor))
	Ic = torch.gather(im_flat, dim=0, index=idx_c.type(torch.cuda.LongTensor))
	Id = torch.gather(im_flat, dim=0, index=idx_d.type(torch.cuda.LongTensor))

	warped_flat = (wa * Ia) + (wb * Ib) + (wc * Ic) + (wd * Id)
	warped = torch.reshape(warped_flat,(num_batch,height,width,channels))
	warped = warped.permute(0,3,1,2)
	im = im.permute(0,3,1,2)
	flow = flow.permute(0,3,1,2)


	return warped

