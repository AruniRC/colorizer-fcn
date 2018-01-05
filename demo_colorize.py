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



data_root = '/vis/home/arunirc/data1/datasets/ImageNet/images/'
exp_folder = '/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn8s_color_CFG-019_VCS-5ec8e90_TIME-20180103-182032'
GMM_PATH = osp.join(exp_folder, 'gmm.pkl')
MEAN_L_PATH = osp.join(exp_folder, 'mean_l.npy')
MODEL_PATH = osp.join(exp_folder, 'model_best.pth.tar')
cuda = torch.cuda.is_available()



def main():

    dataset = data_loader.ColorizeImageNet(
                data_root, split='val', set='small',
                bins='soft', num_hc_bins=16,
                gmm_path=GMM_PATH, mean_l_path=MEAN_L_PATH)

    model = models.FCN8sColor(n_class=16, bin_type='soft')
    checkpoint = torch.load(MODEL_PATH)        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if not osp.exists(osp.join(exp_folder, 'colorized-output-max')):
        os.makedirs(osp.join(exp_folder, 'colorized-output-max'))


    for i in xrange(100):

    	print i

        input_im, labels = dataset.__getitem__(i)
        gmm = dataset.gmm
        mean_l = dataset.mean_l

        # Original RGB image
        img_file = dataset.files['val'][i]
        im_orig = PIL.Image.open(img_file)
        if len(im_orig.size)!=2:
            continue

        if len(im_orig.getbands()) != 3:
            continue
            
        im_orig = dataset.rescale(im_orig)

        # ground-truth GMM posteriors
        labels = labels.numpy()
        img = input_im.numpy().squeeze()
        im_rgb = utils.colorize_image_hc(labels, img, gmm, mean_l)


        # predictions from colorizer network
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
        im_pred = utils.colorize_image_hc(preds, img, gmm, mean_l, 
        								  method='max')

        tiled_img = np.concatenate(
                    (im_orig, 
                     np.zeros([im_rgb.shape[0],10,3], dtype=np.uint8),
                     im_rgb, 
                     np.zeros([im_rgb.shape[0],10,3], dtype=np.uint8), 
                     im_pred), axis=1)

        out_im_file = osp.join(exp_folder, 'colorized-output-max', str(i)+'.jpg')
        skimage.io.imsave(out_im_file, tiled_img)


    # tiled_img = skimage.img_as_ubyte(tiled_img)    





if __name__ == '__main__':
    main()
