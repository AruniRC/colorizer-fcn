import argparse
import os
import os.path as osp
import numpy as np
import PIL.Image
import skimage.io
import skimage.color as color
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import utils
import data_loader
import models



# data_root = '/vis/home/arunirc/data1/datasets/ImageNet/images/'
# exp_folder = '/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn8s_color_CFG-019_VCS-5ec8e90_TIME-20180103-182032'
# GMM_PATH = osp.join(exp_folder, 'gmm.pkl')
# MEAN_L_PATH = osp.join(exp_folder, 'mean_l.npy')
# MODEL_PATH = osp.join(exp_folder, 'model_best.pth.tar')
# binning = 'soft'

# USER: modify settings here
binning = 'uniform'
train_set = 'bright-1'  # {'bright-1': colorful images}
data_root = '/vis/home/arunirc/data1/datasets/ImageNet/images/'
code_root = '/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn'
exp_folder = osp.join(code_root, 'logs', 'MODEL-fcn8s_color_CFG-030_TIME-20180122-230502')
MEAN_L_PATH = osp.join(code_root, 'logs',
                'MODEL-fcn32s_color_CFG-028_TIME-20180118-180822/mean_l.npy')
MODEL_PATH = osp.join(exp_folder, 'model_best.pth.tar')
split = 'train'


cuda = torch.cuda.is_available()


def main():

    # -----------------------------------------------------------------------------
    #   Setup
    # -----------------------------------------------------------------------------
    if binning == 'soft':
        dataset = data_loader.ColorizeImageNet(
                    data_root, split='val', set='small',
                    bins='soft', num_hc_bins=16,
                    gmm_path=GMM_PATH, mean_l_path=MEAN_L_PATH)
        model = models.FCN8sColor(n_class=16, bin_type='soft')

    elif binning == 'uniform':
        dataset = data_loader.ColorizeImageNet(
            data_root, split=split, 
            bins=binning, num_hc_bins=256, 
            set=train_set, im_size=(256, 256),
            gmm_path=None, mean_l_path=MEAN_L_PATH, 
            uniform_sigma='default')
        model = models.FCN8sColor(n_class=256, bin_type='uniform')

    else:
        raise NotImplementedError

    checkpoint = torch.load(MODEL_PATH)        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if not osp.exists(osp.join(exp_folder, 'colorized-output-max-'+split)):
        os.makedirs(osp.join(exp_folder, 'colorized-output-max-'+split))

    # -----------------------------------------------------------------------------
    #   Colorize 100 images from dataset
    # -----------------------------------------------------------------------------
    for i in xrange(100):
        i = i + 10000
    	print i

        input_im, labels = dataset.__getitem__(i)
        gmm = dataset.gmm
        mean_l = dataset.mean_l

        # Original RGB image
        img_file = dataset.files[split][i]
        im_orig = PIL.Image.open(img_file)
        if len(im_orig.size)!=2:
            continue

        if len(im_orig.getbands()) != 3:
            continue
            
        im_orig = dataset.rescale(im_orig)

        # "Ground-truth" colorization
        labels = labels.numpy()
        img = input_im.numpy().squeeze()
        im_rgb = utils.colorize_image_hc(labels, img, gmm, mean_l)


        # Get Hue-Chroma bin predictions from colorizer network
        input_im = input_im.unsqueeze(0)
        inputs = Variable(input_im)
        if cuda:
            model.cuda()
            inputs = inputs.cuda()

        outputs = model(inputs)
        outputs = F.softmax(outputs)
        preds = outputs.squeeze()
        preds = preds.permute(1,2,0)
        preds = preds.data.cpu().numpy()

        # Get a colorized image using predicted Hue-Chroma bins
        im_pred = utils.colorize_image_hc(
                    preds, img, gmm, mean_l, 
        			method='max')

        tiled_img = np.concatenate(
                    (im_orig, 
                     np.zeros([im_rgb.shape[0],10,3], dtype=np.uint8),
                     im_rgb, 
                     np.zeros([im_rgb.shape[0],10,3], dtype=np.uint8), 
                     im_pred), axis=1)

        out_im_file = osp.join(exp_folder, 'colorized-output-max-'+split, 
                               str(i)+'.jpg')
        skimage.io.imsave(out_im_file, tiled_img)


    # tiled_img = skimage.img_as_ubyte(tiled_img)    





if __name__ == '__main__':
    main()
