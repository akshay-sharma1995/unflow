import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from util import conv, predict_flow, deconv, crop_like, correlate
import skimage.io
from matplotlib import pyplot as plt
import numpy as np

class FlowNetC(nn.Module):
    def __init__(self,batchNorm=True):
        super(FlowNetC,self).__init__()

        self.batchNorm = batchNorm
        self.conv1      = conv(self.batchNorm,   1,   64, kernel_size=7, stride=2)
        self.conv2      = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3      = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256,   32, kernel_size=1, stride=1)

        self.conv3_1 = conv(self.batchNorm, 473,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):# X.SHAPE = Bx2xcxhxw
        x1 = torch.tensor(x[:,0]).type(torch.cuda.FloatTensor) ## x1.shape = B,C,H,W
        # print("x.shape",np.shape(x1))
        x2 = torch.tensor(x[:,1]).type(torch.cuda.FloatTensor)
        # x3 = torch.tensor(x[:,2]).type(torch.cuda.FloatTensor)
        # x1 = np.rollaxis(x1,2,1)
        # x1 = np.rollaxis(x1,3,2)
        # skimage.io.imshow(x1[0])
        # plt.show()

        out_conv1a = self.conv1(x1)
        
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)
        out_conv_redir = self.conv_redir(out_conv3a)
        out_correlation = correlate(out_conv3a,out_conv3b)



        in_conv3_1 = torch.cat([out_conv_redir, out_correlation], dim=1)

        out_conv3 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))


        flow6       = self.predict_flow6(out_conv6)
        
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2a)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2a)

        concat2 = torch.cat((out_conv2a,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        # print("mnsadkandn")
        # print("shape",flow2.size(),flow3.size(),flow4.size(),flow5.size(),flow6.size())
        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2
