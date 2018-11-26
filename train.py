import torch
import torch.nn.functional as F
import numpy as np
import time
import skimage.io
from losses import *
from matplotlib import pyplot as plt

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
        out_image.append(F.grid_sample(train_loader[:,0],ofnp))

    loss_image = loss(train_loader[:,1],out_image)

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