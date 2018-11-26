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
# import models
# import datasets
# from multiscaleloss import multiscaleEPE, realEPE
# import datetime
# from tensorboardX import SummaryWriter
import numpy as np
import random
from matplotlib import pyplot as plt
from FlownetC import *
import torch.optim as optim
import torch
from train import *

def imlist(fpath):
    flist = os.listdir(fpath)
    return flist

def channel_rearrange(image_batch):
    return


def loader():
    model = FlowNetC()
    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    DIR = "../../../Project_data/training_data/small_dataset_npz_3_set/"
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])
    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0],std=[20,20])
    ])
    batch_size = 10
    # length =
    flist = imlist(DIR)
    # print("flist",flist)
    number_of_image_sets = len(flist)
    idxs = (np.arange(1,number_of_image_sets,1)) ## id of all image_sets
    random.shuffle(idxs) ## shuffling the idxs

    for i in range(len(idxs)):
        image_batch = []

        for j in range(10): ## making batches
            path = DIR + flist[idxs[i]]
            image_triplet = np.load(path)['arr_0']
            image_batch.append(image_triplet)
        image_batch = np.array(image_batch)
        image_batch = np.rollaxis(image_batch,4,2)

        # print("listofimages.shape",np.shape(image_batch))
        train(image_batch,model,optimizer)

def main():
    loader()
if __name__ == '__main__':
    main()
