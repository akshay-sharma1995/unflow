import torch
import torch.nn.functional as F
import numpy as np
import time
import skimage.io
from losses import *
from matplotlib import pyplot as plt
import cv2
def display_flow(u,v,x_orig):
    hsv = np.zeros(x_orig.size(), dtype=np.uint8)
    hsv = np.rollaxis(np.rollaxis(hsv,1,0),2,1)
    print("hsv",hsv.shape)
    hsv[..., 1] = 255
    print("hsv",hsv.shape)
    print("u.shape",u.shape)
    mag, ang = cv2.cartToPolar(u, v)
    print("mag.shape",np.shape(mag),np.shape(ang))
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow("colored flow", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def train(train_loader, model, optimizer):
    # global n_iter, args
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # flow2_EPEs = AverageMeter()

    # epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)
    epoch_size = 1
    epoch = 0
    # switch to train mode
    train_loader = torch.tensor(train_loader).type(torch.cuda.FloatTensor)
    model.train()

    end = time.time()

    out_flow = model(train_loader)
    h, w = train_loader.size()[-2:]
    # print(h,w)
    out_flow = [F.interpolate(oflow, (h,w)) for oflow in out_flow]
    # print("out_flow.shape",out_flow[1].size(),   train_loader.shape)
    # fl = out_flow[0].numpy()
    out_image = []

    for p in range(0,5):
        ofnp = out_flow[p].permute(0,2,3,1)
        flow_h = ofnp.size()[1]
        print('flow_h',flow_h)
        flow_w = ofnp.size()[2]
        ofnp[:,0:1] = ofnp[:,0:1]/flow_h
        # if(ofnp[:,0:1]>1):
        #      ofnp[:,0:1]=1
        ofnp[:,1:2] = ofnp[:,1:2]/flow_w 
        # ofnp = np.meshgrid(np.linspace(0,0,flow_w), np.linspace(-0,0,flow_h))
        # ofnp = torch.tensor(ofnp).type(torch.cuda.FloatTensor)
        # ofnp = ofnp.unsqueeze(0)
        # ofnp = ofnp.permute(0,2,3,1)
        # print('ofnp.sahpe',ofnp.size())

        # ofnp = torch.clamp(ofnp,min=-0.0,max=0.0)
        # ofnp = (ofnp/torch.max(ofnp)[0])*2-1
        # ofnp =  
        print('Max',torch.max(ofnp))
        print('Min',torch.min(ofnp))
        out_image.append(F.grid_sample((train_loader[:,0]+1)/2,ofnp))

    

    x_orig = train_loader[0,0]
    x_orig = 255.0*(x_orig)     
    x_orig = x_orig.type(torch.cuda.IntTensor)
    skimage.io.imsave("./orginial_img.png",np.rollaxis(np.rollaxis(x_orig.cpu().data.numpy(),1,0),2,1))
    
    xi = out_image[0][0,:]
    xi = 255.0*(xi + 1.0)/2.0    
    xi = xi.type(torch.cuda.IntTensor)
    skimage.io.imsave("./warped_img.png",np.rollaxis(np.rollaxis(xi.cpu().data.numpy(),1,0),2,1))
    # plt.show()
    
    u = out_flow[0][0,0].cpu().data.numpy()
    v = out_flow[0][0,1].cpu().data.numpy()
    # u = np.rollaxis(np.rollaxis(out_flow[0][0,0].cpu().data.numpy(),1,0),2,1)
    # v = np.rollaxis(np.rollaxis(out_flow[0][0,1].cpu().data.numpy(),1,0),2,1)
    display_flow(u,v,x_orig)
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


    end
    loss_image = loss(train_loader[:,1],out_image,out_flow[0])

    print("loss_image",loss_image.data)
    optimizer.zero_grad()
    loss_image.backward()
    optimizer.step()


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