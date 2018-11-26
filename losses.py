import torch
import torch.nn.functional as F

def loss(target,out_image):
    criterion = torch.nn.MSELoss()
    weights = [0.005, 0.01, 0.02, 0.08, 0.32]
    loss = 0
    for i in range(5):
        loss+= weights[i]*criterion(target,out_image[i])
    return loss


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
        # self.layer1 = nn.
    def forward(self, X, Y, Z):
    	hf1 = torch.tensor([[0,0,0],[-1,2,-1],[0,0,0]])
    	hf2 = torch.tensor([[0,-1,0],[0,2,0],[0,-1,0]])
    	hf3 = torch.tensor([[-1,0,-1],[0,4,0],[-1,0,-1]])
        # diff = torch.add(X, -Y)
        
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error) 
return loss 