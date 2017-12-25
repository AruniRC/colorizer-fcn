import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import skimage.color as color
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib
import torch
from torch.utils import data

# DEBUG_DIR = '/data2/arunirc/Research/colorize-fcn/pytorch-fcn/tests/data_tests'

class ColorizeImageNet(data.Dataset):
    '''
        Dataset subclass for learning colorization on ImageNet data.
        Each image is transformed into HSV, then to H, L and C (chroma).
        L is the single-channel input to a network.
        The network predicts target histograms of H and C, computed over 
        a 7x7 window around each target, 32 bins for each. Axes are binned 
        uniformly in [0,1].

        Folder structure:
            .../ImageNet/images/
                |-- train/
                |-- val/
                |-- files_100k_train.txt
                |-- files_100k_val.txt
                |-- untarScript.sh

        Example 1
        ---------
        from color_imagenet import *
        root = '/data/arunirc/datasets/ImageNet/images/'
        dataset = ColorizeImageNet(root, split='val', set='small')
        print 'Number of records: %d' % dataset.__len__()
        img, lbl = dataset.__getitem__(0)
        assert len(lbl)==2
        im_hue = lbl[0].numpy()
        im_chroma = lbl[1].numpy()
        assert img.shape==im_hue.shape
        assert img.shape==im_chroma.shape

        Example 2
        ---------
        # Check for 'train' split, data ranges
        root = '/data/arunirc/datasets/ImageNet/images/'
        dataset = ColorizeImageNet(root, split='train', set='small')
        img, lbl = dataset.__getitem__(0)
        assert np.min(lbl[0].numpy())==1
        assert np.max(lbl[0].numpy())==31

        Example 3
        ---------
        # Check operation as an Iterable
        root = '/data/arunirc/datasets/ImageNet/images/'
        train_loader = torch.utils.data.DataLoader(
                        torchfcn.datasets.ColorizeImageNet(root, split='train'),
                        batch_size=1, shuffle=True)
        img, label = dataiter.next()
        assert len(img.size()) == 4
        assert type(img)==torch.FloatTensor

    '''

    # -----------------------------------------------------------------------------
    def __init__(self, root, log_dir = '.', split='train', set='small', \
                             num_hc_bins=32, bins='one-hot'):
    # -----------------------------------------------------------------------------
        '''
            Parameters
            ----------
            root        -   Path to root of ImageNet dataset
            log_dir     -   Output location to cache results (needed for soft-binning)
            split       -   Either 'train' or 'val'
            set         -   Can be 'full', 'small' (100k samples) or 'tiny' (9 images)
            num_hc_bins -
            bins        -   String. If 'one-hot' - bins separately for Hue and
                            Chroma channels. Values quantized uniformly in 
                            [0..1]. If 'soft', then a GMM is used to assign soft 
                            binning to each pixel in a [1xnum_hc_bins] vector. 
                            In the 'soft' case, KLDivLoss loss needs to be used
                            in the network, instead of the usual log-loss.
        ''' 
        self.root = root  # '.../ImageNet/images'
        self.log_dir = log_dir
        self.split = split
        self.files = collections.defaultdict(list)
        self.num_hc_bins = num_hc_bins
        self.hc_bins = np.linspace(0,1,num=num_hc_bins) # Hue and chroma bins (fixed)
        self.im_size = 500.0 # scale larger side of image to this value
        self.bins = bins
        self.set = set
        self.gmm = []

        if set == 'small':
            # A toy dataset of 100k samples for rapid prototyping
            files_small = osp.join(root, 'files_100k_'+split+'.txt')
            assert osp.exists(files_small), 'File does not exist: %s' % files_small
            imfn = []
            with open(files_small, 'r') as ftrain:
                for line in ftrain:
                    imfn.append(osp.join(root, line.strip()))
            self.files[split] =  imfn

        elif set == 'tiny':
            # DEBUG: 9 val set images
            files_small = osp.join(root, 'tiny_val.txt')
            assert osp.exists(files_small), 'File does not exist: %s' % files_small
            imfn = []
            with open(files_small, 'r') as ftrain:
                for line in ftrain:
                    imfn.append(osp.join(root, line.strip()))
            self.files[split] =  imfn


        # TODO - read in entire ImageNet dataset filenames


        if self.bins=='soft':
            # estimate GMM on joint Hue and Chroma values
            #   1. sample data points (num_images*pixel_subset)
            #   2. fit GMM
            gmm_path = osp.join(self.log_dir, 'gmm.pkl')

            if osp.exists(gmm_path):
                print 'Loading GMM parameters from cache'
                self.gmm = joblib.load(gmm_path)

            else:
                color_samples = self.get_color_samples(num_images=1000, pixel_subset=20)
                gmm = GaussianMixture(n_components=num_hc_bins, \
                                      covariance_type='full', init_params='kmeans', \
                                      random_state=0, verbose=1)
                gmm.fit(color_samples)
                self.gmm = gmm
                joblib.dump(gmm, gmm_path)
                print 'done GMM fitting'




    # -----------------------------------------------------------------------------
    def __len__(self):
    # -----------------------------------------------------------------------------
        return len(self.files[self.split])


    # -----------------------------------------------------------------------------
    def __getitem__(self, index):
    # -----------------------------------------------------------------------------
        img_file = self.files[self.split][index]
        img = PIL.Image.open(img_file)

        # HACK: for non-RGB images - 4-channel CMYK or 1-channel grayscale
        if len(img.getbands()) != 3:
            while len(img.getbands()) != 3:
                index -= 1
                img_file = self.files[self.split][index] # if -1, wrap-around
                img = PIL.Image.open(img_file)

        # scale largest side to 500 px, maintain aspect ratio
        scale_factor = self.im_size/np.max(img.size)
        if scale_factor != 1:
            scaled_dim = ( np.round(scale_factor*img.size[0]).astype(np.int32), \
                          np.round(scale_factor*img.size[1]).astype(np.int32) )
            img = img.resize(scaled_dim, PIL.Image.BILINEAR)

        img = np.array(img, dtype=np.uint8)

        # Colorspace conversion: 
        #   RGB --> HSV; HSV --> H (hue), C (chroma), L (lightness)
        h, c, L = self.rgb_to_hue_chroma(img)
        # TODO - zero center
        # L = L * 255.0 # HACK: expand values

        # Binning: one-hot (labels and log-loss), soft (soft-binning and KL-div)
        if self.bins == 'one-hot':
            h_label = self.make_label_map(h) # 1 x H x W
            c_label = self.make_label_map(c)
            hc_label = (torch.from_numpy(h_label).long(), \
                        torch.from_numpy(c_label).long())
        elif self.bins == 'soft':
            hc_label = torch.from_numpy(self.make_soft_bin_map(h, c)).float()
            # assert hc_label.shape == (L.shape[0], L.shape[1], self.num_hc_bins)

        # Lightness channel (L) is the single-channel input to the network
        im_out = np.expand_dims(L, axis=0) # 1 x H x W
        im_out = torch.from_numpy(im_out).float()
        return im_out, hc_label


    # -----------------------------------------------------------------------------
    def make_label_map(self, im_hc):
    # -----------------------------------------------------------------------------
        hc_label = np.digitize(im_hc, self.hc_bins, right=False)
        hc_label = hc_label.astype(np.int32) - 1 # index 0 .. hc_bins-1
        assert np.min(hc_label)>=0, 'Negative label value.'
        return hc_label


    # -----------------------------------------------------------------------------
    def make_soft_bin_map(self, im_h, im_c):
    # -----------------------------------------------------------------------------
        hc_samples = np.stack( (im_h.flatten(), im_c.flatten()), axis=1 )
        gmm_posteriors = self.gmm.predict_proba(hc_samples)        
        bin_map = gmm_posteriors.reshape( \
                    (im_h.shape[0], im_h.shape[1], self.num_hc_bins) )
        return bin_map


    # -----------------------------------------------------------------------------
    def rgb_to_hue_chroma(self, img):
    # -----------------------------------------------------------------------------
        im_hsv = color.rgb2hsv(img)
        if np.isnan(np.sum(im_hsv)):
            raise ValueError('HSV from RGB conversion has NaN.')
        h = im_hsv[:,:,0]
        s = im_hsv[:,:,1]
        v = im_hsv[:,:,2]
        c = v * s
        L = v - (c/2.0)
        return h, c, L


    # -----------------------------------------------------------------------------
    def transform(self, img, lbl):
    # -----------------------------------------------------------------------------
        # img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        # img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


    # -----------------------------------------------------------------------------
    def untransform(self, img):
    # -----------------------------------------------------------------------------
        scale_factor = self.im_size/np.max(img.size) # img is PIL.Image
        if scale_factor != 1:
            scaled_dim = ( np.round(scale_factor*img.size[0]).astype(np.int32), \
                          np.round(scale_factor*img.size[1]).astype(np.int32) )
            img = img.resize(scaled_dim, PIL.Image.BILINEAR)
        img = np.array(img, dtype=np.uint8) # PIL.Image to numpy array
        return img


    # -----------------------------------------------------------------------------
    def get_color_samples(self, num_images=1000, pixel_subset=50):
    # -----------------------------------------------------------------------------
        '''
            Returns an Nx2 vector of sampled hue and chroma values, which can be 
            used learn a Gaussian Mixture Model. N = num_images*pixel_subset. 
            This method depends on having a 100k subset of train files from the
            ImageNet dataset available as a text-file (`files_small`).
        '''
        if self.set == 'tiny':
            files_small = osp.join(self.root, 'tiny_val.txt')
        else:
            files_small = osp.join(self.root, 'files_100k_train.txt')
        assert osp.exists(files_small), 'File does not exist: %s' % files_small

        # random subset of `num_images`
        imfn = []
        with open(files_small, 'r') as ftrain:
            for line in ftrain:
                imfn.append(osp.join(self.root, line.strip()))

        num_images = np.min([num_images, len(imfn)])
        if num_images < 100:
            pixel_subset = 1000

        sel = np.random.randint(0, np.min([num_images, len(imfn)]), num_images)
        imfn_subset = [imfn[i] for i in sel] # subset of filenames

        hc_sample = []
        print 'sampling pixels for Hue and Chroma . . . \r'
        for img_file in imfn_subset:
            im = PIL.Image.open(img_file)
            if len(im.getbands()) != 3:
                continue
            im = np.array(im, dtype=np.uint8)
            im_h, im_c, _ = self.rgb_to_hue_chroma(im)
            im_h = im_h.flatten()
            im_c = im_c.flatten()
            sel_pixel = np.random.randint(0, len(im_h), pixel_subset)
            hc_sample.append(np.stack((im_h[sel_pixel], im_c[sel_pixel]), 1))

        hc_sample = np.concatenate(hc_sample, 0)
        print 'finished sampling: %d' % hc_sample.shape[0]
        return hc_sample





    # -----------------------------------------------------------------------------
    def pred_to_hc(self, pred_hue, pred_chroma):
    # -----------------------------------------------------------------------------
        '''
            TODO
            Convert the predicted labels of Hue and Chroma into actual values, 
            using bin centers.

            Binning:  bins[i-1] <= x < bins[i] 

            Parameters
            ----------
            pred_hue    - Predicted Hue map, integer labels. 
            pred_chroma - Predicted Chroma map, integer labels. 
    
            Returns
            -------
            im_hue, im_chroma

        '''
        bin_centers = (self.hc_bins[:-1] + self.hc_bins[1:])/2.0