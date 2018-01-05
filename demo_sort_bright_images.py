import argparse
import os
import os.path as osp
import numpy as np
import PIL.Image
import skimage.io
import skimage.color as color
import skimage.io as io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import utils
import data_loader
import models



data_root = '/vis/home/arunirc/data1/datasets/ImageNet/images/'
exp_folder = osp.join('logs', 'sorted-rgbvar-images')
method = 'rgbvar'


def main():

    dataset = data_loader.ColorizeImageNet(
                data_root, split='val', set='small', bins='one-hot')

    if not osp.exists(exp_folder):
        os.makedirs(exp_folder)

    im_satval = []
    im_filenames = []
    for i in xrange(100):

    	print i

        # Original RGB image
        img_file = dataset.files['val'][i]
        im_orig = PIL.Image.open(img_file)

        # Invalid image formats
        if len(im_orig.size)!=2:
            continue

        if len(im_orig.getbands()) != 3:
            continue
            
        im_orig = dataset.rescale(im_orig)
        im_hsv = skimage.color.rgb2hsv(im_orig)
        im_filenames.append(img_file)

        if method == 'saturation':
            im_satval.append(im_hsv[:,:,1].mean())
        elif method == 'chroma':
            chroma = im_hsv[:,:,1] * im_hsv[:,:,2]
            im_satval.append(chroma.mean())
        elif method == 'rgbvar':
            rgb_var = im_orig.var(axis=2)
            im_satval.append(rgb_var.mean())
        else:
            raise ValueError

    # sort: greatest value first
    sorted_idx = np.argsort(im_satval)[::-1]

    for i in xrange(len(sorted_idx)):
        print im_satval[sorted_idx[i]]
        img_file = im_filenames[sorted_idx[i]]
        im_orig = skimage.io.imread(img_file)
        out_im_file = osp.join(exp_folder, 
                        '{0:0{width}}'.format(i, width=6) + \
                        '_' + str(im_satval[sorted_idx[i]]) + '.jpg')
        skimage.io.imsave(out_im_file, im_orig)


if __name__ == '__main__':
    main()
