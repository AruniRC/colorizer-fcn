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
import tqdm
from subprocess import call

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import utils
import data_loader
import models



data_root = '/vis/home/arunirc/data1/datasets/ImageNet/images/'
exp_folder = osp.join('logs', 'sorted-rgbvar-imagenet')
method = 'rgbvar'
batch_sz = 128





def main():

    # sort all ImageNet images based on "bright" colors: var(r,g,b)
    if not osp.exists(exp_folder):
        sort_bright_images()
   
    print batch_sz

    # TODO - save a few images
    sorted_filenames = np.load(osp.join(exp_folder, 'files_sorted.npy'))
    sorted_vals = np.load(osp.join(exp_folder, 'values_sorted.npy'))

    # plot values
    f = plt.figure()
    plt.plot(sorted_vals)
    plt.ylabel('var(r,g,b)')
    plt.xlabel('sorted ImageNet images')
    plt.tight_layout()
    plt.savefig(osp.join(exp_folder, 'sorted_values.png'), bbox_inches='tight')


    # first 100
    if not osp.exists(osp.join(exp_folder, 'first-100')):
        os.makedirs(osp.join(exp_folder, 'first-100'))

        for i in tqdm.trange(100):
            im = PIL.Image.open(sorted_filenames[i])
            im.save(osp.join(exp_folder, 'first-100', str(i)+'.jpg'))

    # 1k
    if not osp.exists(osp.join(exp_folder, '1k-100')):
        os.makedirs(osp.join(exp_folder, '1k-100'))

        for i in tqdm.trange(100):
            im = PIL.Image.open(sorted_filenames[i+1000])
            im.save(osp.join(exp_folder, '1k-100', str(i)+'.jpg'))


    # 10k
    if not osp.exists(osp.join(exp_folder, '10k-100')):
        os.makedirs(osp.join(exp_folder, '10k-100'))

        for i in tqdm.trange(100):
            im = PIL.Image.open(sorted_filenames[i+10000])
            im.save(osp.join(exp_folder, '10k-100', str(i)+'.jpg'))

    # 100k
    if not osp.exists(osp.join(exp_folder, '100k-100')):
        os.makedirs(osp.join(exp_folder, '100k-100'))

        for i in tqdm.trange(100):
            im = PIL.Image.open(sorted_filenames[i+100000])
            im.save(osp.join(exp_folder, '100k-100', str(i)+'.jpg'))


    if not osp.exists(osp.join(exp_folder, '200k-100')):
        os.makedirs(osp.join(exp_folder, '200k-100'))

        for i in tqdm.trange(100):
            im = PIL.Image.open(sorted_filenames[i+200000])
            im.save(osp.join(exp_folder, '200k-100', str(i)+'.jpg'))

        # os.system('cd ./logs/sorted-rgbvar-imagenet/200k-100')
        # os.system('montage `ls` montage-200k-100.jpg')
        # os.system('mv montage-200k-100.jpg ../montage-200k-100.jpg')



    if not osp.exists(osp.join(exp_folder, '300k-100')):
        os.makedirs(osp.join(exp_folder, '300k-100'))

        for i in tqdm.trange(100):
            im = PIL.Image.open(sorted_filenames[i+300000])
            im.save(osp.join(exp_folder, '300k-100', str(i)+'.jpg'))

        # os.system('cd ./logs/sorted-rgbvar-imagenet/300k-100')
        # os.system('montage `ls` montage-300k-100.jpg')
        # os.system('mv montage-300k-100.jpg ../montage-300k-100.jpg')


    if not osp.exists(osp.join(exp_folder, '500k-100')):
        os.makedirs(osp.join(exp_folder, '500k-100'))

        for i in tqdm.trange(100):
            im = PIL.Image.open(sorted_filenames[i+500000])
            im.save(osp.join(exp_folder, '500k-100', str(i)+'.jpg'))

        # os.system('cd ./logs/sorted-rgbvar-imagenet/500k-100')
        # os.system('montage `ls` montage-500k-100.jpg')
        # os.system('mv montage-500k-100.jpg ../montage-500k-100.jpg')


    if not osp.exists(osp.join(exp_folder, '600k-100')):
        os.makedirs(osp.join(exp_folder, '600k-100'))

        for i in tqdm.trange(100):
            im = PIL.Image.open(sorted_filenames[i+600000])
            im.save(osp.join(exp_folder, '600k-100', str(i)+'.jpg'))

        # os.system('cd ./logs/sorted-rgbvar-imagenet/600k-100')
        # os.system('montage `ls` montage-600k-100.jpg')
        # os.system('mv montage-600k-100.jpg ../montage-600k-100.jpg')


    if not osp.exists(osp.join(exp_folder, '700k-100')):
        os.makedirs(osp.join(exp_folder, '700k-100'))

        for i in tqdm.trange(100):
            im = PIL.Image.open(sorted_filenames[i+700000])
            im.save(osp.join(exp_folder, '700k-100', str(i)+'.jpg'))

        # os.system('cd ./logs/sorted-rgbvar-imagenet/700k-100')
        # os.system('montage `ls` montage-700k-100.jpg')
        # os.system('mv montage-700k-100.jpg ../montage-700k-100.jpg')

    if not osp.exists(osp.join(exp_folder, '900k-100')):
        os.makedirs(osp.join(exp_folder, '900k-100'))

        for i in tqdm.trange(100):
            im = PIL.Image.open(sorted_filenames[i+900000])
            im.save(osp.join(exp_folder, '900k-100', str(i)+'.jpg'))

        # os.system('cd ./logs/sorted-rgbvar-imagenet/900k-100')
        # os.system('montage `ls` montage-900k-100.jpg')
        # os.system('mv montage-900k-100.jpg ../montage-900k-100.jpg')



def sort_bright_images():
    if not osp.exists(exp_folder):
        os.makedirs(exp_folder)

    transform = transforms.Compose([
                transforms.Scale((256, 256)),
                transforms.ToTensor()])

    datadir = osp.join(data_root, 'train')

    dataset = datasets.ImageFolder(datadir, transform)
    data_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_sz, shuffle=False)

    dataiter = iter(data_loader)

    im_satval = []
    im_filenames = []

    for i in tqdm.trange(len(data_loader)):

        batch_data, label = dataiter.next()
        tensor2im = torchvision.transforms.ToPILImage()

        # # DEBUG
        # if i==5:
        #     break

        for j in range(batch_data.size()[0]):

            im_orig = tensor2im(batch_data[j]) # PIL Image

            # Invalid image formats
            if len(im_orig.size)!=2:
                continue

            if len(im_orig.getbands()) != 3:
                continue
                
            im_orig = np.array(im_orig, dtype=np.uint8)
            im_hsv = skimage.color.rgb2hsv(im_orig)

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

            img_file = dataset.imgs[j + i*batch_sz][0]
            im_filenames.append(img_file)

    
    im_filenames = np.asarray(im_filenames)
    im_satval = np.asarray(im_satval)

    # sort: greatest value first
    sorted_idx = np.argsort(im_satval)[::-1]
    sorted_filenames = im_filenames[sorted_idx]
    sorted_satvals = im_satval[sorted_idx]

    np.save(osp.join(exp_folder, 'files_sorted.npy'), sorted_filenames)
    np.save(osp.join(exp_folder, 'values_sorted.npy'), sorted_satvals)




if __name__ == '__main__':
    main()