import argparse
import os
import os.path as osp

import numpy as np
import PIL.Image
import skimage.io
import skimage.color as color
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torchfcn


root = '/data/arunirc/datasets/ImageNet/images/'
out_path = '/data2/arunirc/Research/colorize-fcn/pytorch-fcn/tests/data_tests/'

def hc_pred_to_rgb():



def main():

    dataset = torchfcn.datasets.ColorizeImageNet(\
                root, split='train', set='small')
    img_file = dataset.files['train'][100]
    






if __name__ == '__main__':
    main()
