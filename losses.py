import torch
import torch.nn as nn
import torch.nn.functional as F

def loss(target,out_image,flow):
	s_loss_model = spatial_smoothing_loss()
	s_loss = s_loss_model(flow)
	print("s_loss",s_loss)
	criterion = torch.nn.MSELoss()
	weights = [0.005, 0.01, 0.02, 0.08, 0.32]
	loss = 0
	for i in range(1):
		loss+= weights[i]*criterion(target,out_image[i])
	return loss




class spatial_smoothing_loss(nn.Module):
	"""L1 Charbonnierloss."""
	def __init__(self):
		super(spatial_smoothing_loss, self).__init__()
		self.eps = 1e-6
		# self.conv = F.conv2d()
		# self.layer1 = nn.
	def forward(self, X): ## X is flow map
		u = X[:,0:1]
		v = X[:,1:2]
		print("u",u.size())
		hf1 = torch.tensor([[[[0,0,0],[-1,2,-1],[0,0,0]]]]).type(torch.cuda.FloatTensor)
		hf2 = torch.tensor([[[[0,-1,0],[0,2,0],[0,-1,0]]]]).type(torch.cuda.FloatTensor)
		hf3 = torch.tensor([[[[-1,0,-1],[0,4,0],[-1,0,-1]]]]).type(torch.cuda.FloatTensor)
		# diff = torch.add(X, -Y)
		
		u_hloss = F.conv2d(u,hf1,padding=1,stride=1)
		# print("uhloss",type(u_hloss))
		u_vloss = F.conv2d(u,hf2,padding=1,stride=1)
		u_dloss = F.conv2d(u,hf3,padding=1,stride=1)

		v_hloss = F.conv2d(v,hf1,padding=1,stride=1)
		v_vloss = F.conv2d(v,hf2,padding=1,stride=1)
		v_dloss = F.conv2d(v,hf3,padding=1,stride=1)

		u_hloss = charbonier(u_hloss,self.eps)
		u_vloss = charbonier(u_vloss,self.eps)
		u_dloss = charbonier(u_dloss,self.eps)

		v_hloss = charbonier(v_hloss,self.eps)
		v_vloss = charbonier(v_vloss,self.eps)
		v_dloss = charbonier(v_dloss,self.eps)


		# error = torch.sqrt( diff * diff + self.eps )
		# loss = torch.sum(error) 
		loss = u_hloss + u_vloss + u_dloss + v_hloss + v_vloss + v_dloss
		# print('char_losss',loss)
		return loss 

def charbonier(x,eps):
	gamma = 0.45
	# print("x.type",type(x))
	loss = x*x + eps*eps
	loss = torch.pow(loss,gamma)
	loss = torch.mean(loss)
	return loss