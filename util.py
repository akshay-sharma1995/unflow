import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import cv2
import skimage.io
import argparse
import os

try:
    from spatial_correlation_sampler import spatial_correlation_sample
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load custom correlation module"
                      "which is needed for FlowNetC", ImportWarning)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lrd', dest='lr_disc', type=float, default=1e-3, help='learning rate_for_discriminator')
    parser.add_argument('--lrg', dest='lr_gen', type=float, default=1e-3, help='learning rate_for_generator')
    parser.add_argument('--wt-KL', dest='wt_KL', type=float, default=1, help='Weight for KL loss')
    parser.add_argument('--wt-recon', dest='wt_recon', type=float, default=1, help='Weight for Recon loss')
    parser.add_argument('--data-dir', dest='data_dir', type=str, default="../data_scene_flow_multiview/training/image_2/", help='path to data directory')
    parser.add_argument('--data-dir-test', dest='data_dir_test', type=str, default="../data_scene_flow_multiview/testing/image_2/", help='path to test data directory')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--checkpoint', dest='checkpoint_path', type=str, default=None, help='path of a saved_checkpoint')
    parser.add_argument('--train', dest='train', type=int, default=0, help='0 to test the model, 1 to train the model')
    parser.add_argument('--save-interval', dest='save_interval', type=int, default=10, help='epochs after which save the model')
    parser.add_argument('--loaded-epoch', dest='loaded_epoch', type=int, default=1, help='Loading epoch of trained model')
    return parser.parse_args()



def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


def rgb_to_y(input):
  # input is mini-batch N x 3 x H x W of an RGB image
  output = Variable(input.data.new(*input.size()))
  output[:, 0, :, :] = input[:, 0, :, :] * 65.481 + input[:, 1, :, :] * 128.553 + input[:, 2, :, :] * 24.966 + 16
  # similarly write output[:, 1, :, :] and output[:, 2, :, :] using formulas from https://en.wikipedia.org/wiki/YCbCr
  return output[:, 0:1, :, :] ## just the y-channel

def make_dirs(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)

def save_samples(frames,dir_name, epoch, sample, folder_name ):
    # print(frames.shape)
    # pred_frames are numpy ndarray of size (B, 1, H, W)
    save_dir = os.path.join(dir_name, "{}_{}".format(folder_name,epoch))
    make_dirs([save_dir])
    frames = (frames+1)*(255.0/2)
    frames = frames.astype('uint8')
    fname = os.path.join(save_dir, str(sample) + ".png")
    # print(fname)
    skimage.io.imsave(fname,frames)

# def visualize_op_flow(flow):
#   h,w = np.shape(flow)[0],np.shape(flow)[1]
#   hsv = np.zeros((h,w,3), dtype=np.uint8)
#   hsv[..., 1] = 255

#   mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#   hsv[..., 0] = ang * 180 / np.pi / 2
#   hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#   bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#   return bgr`