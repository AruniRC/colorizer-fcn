import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess
import pytz
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

import models
import train
import utils
import data_loader
from config import configurations



here = osp.dirname(osp.abspath(__file__))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('-b', '--binning', default='soft', 
                        choices=('soft','one-hot'))
    parser.add_argument('-k', '--numbins', type=int, default=32)
    parser.add_argument('-d', '--dataset_path', 
                        default='/vis/home/arunirc/data1/datasets/ImageNet/images/')
    parser.add_argument('-m', '--model_path', default=None)
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    cfg = configurations[args.config]
    cfg.update({'bin_type':args.binning,'numbins':args.numbins})
    out = get_log_dir('fcn32s_color', args.config, cfg, verbose=False)
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True     


    # -----------------------------------------------------------------------------
    # 1. dataset
    # -----------------------------------------------------------------------------
    # root = osp.expanduser('~/data/datasets')
    root = args.dataset_path
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    
    if 'img_lowpass' in cfg.keys():
        img_lowpass = cfg['img_lowpass']
    else:
        img_lowpass = None
    if 'train_set' in cfg.keys():
        train_set = cfg['train_set']
    else:
        train_set = 'full'
    if 'val_set' in cfg.keys():
        val_set = cfg['val_set']
    else:
        val_set = 'small'

    if 'gmm_path' in cfg.keys():
        gmm_path = cfg['gmm_path']
    else:
        gmm_path = None
    if 'mean_l_path' in cfg.keys():
        mean_l_path = cfg['mean_l_path']
    else:
        mean_l_path = None
    if 'im_size' in cfg.keys():
        im_size = cfg['im_size']
    else:
        im_size = (256, 256)

    
    # DEBUG: set='tiny'
    train_loader = torch.utils.data.DataLoader(
        data_loader.ColorizeImageNet(root, split='train', 
        bins=args.binning, log_dir=out, num_hc_bins=args.numbins, 
        set=train_set, img_lowpass=img_lowpass, im_size=im_size,
        gmm_path=gmm_path, mean_l_path=mean_l_path),
        batch_size=1, shuffle=True, **kwargs) # DEBUG: set shuffle False

    # DEBUG: set='tiny'
    val_loader = torch.utils.data.DataLoader(
        data_loader.ColorizeImageNet(root, split='val', 
        bins=args.binning, log_dir=out, num_hc_bins=args.numbins, 
        set=val_set, img_lowpass=img_lowpass, im_size=im_size,
        gmm_path=gmm_path, mean_l_path=mean_l_path),
        batch_size=1, shuffle=False, **kwargs)


    # -----------------------------------------------------------------------------
    # 2. model
    # -----------------------------------------------------------------------------
    model = models.FCN32sColor(n_class=args.numbins, bin_type=args.binning)
    if args.model_path:
        checkpoint = torch.load(args.model_path)        
        model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        pass
    if cuda:
        model = model.cuda()


    # -----------------------------------------------------------------------------
    # 3. optimizer
    # -----------------------------------------------------------------------------
    params = filter(lambda p: p.requires_grad, model.parameters())
    if 'optim' in cfg.keys():
    	if cfg['optim'].lower()=='sgd':
    		optim = torch.optim.SGD(params,
				        lr=cfg['lr'],
				        momentum=cfg['momentum'],
				        weight_decay=cfg['weight_decay'])
    	elif cfg['optim'].lower()=='adam':
    		optim = torch.optim.Adam(params,
				        lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    	else:
    		raise NotImplementedError('Optimizers: SGD or Adam')
    else:
	    optim = torch.optim.SGD(params,
			        lr=cfg['lr'],
			        momentum=cfg['momentum'],
			        weight_decay=cfg['weight_decay'])

    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])


    # -----------------------------------------------------------------------------
    # Sanity-check: forward pass with a single sample
    # -----------------------------------------------------------------------------
    DEBUG = True
    if DEBUG:
        dataiter = iter(val_loader)
        img, label = dataiter.next()
        model.eval()
        print 'Labels: ' + str(label.size()) # batchSize x num_class
        print 'Input: ' + str(img.size())    # batchSize x 1 x (im_size)
        if val_loader.dataset.bins == 'one-hot':
            from torch.autograd import Variable
            inputs = Variable(img)
            if cuda:
                inputs = inputs.cuda()
            outputs = model(inputs)
            assert len(outputs)==2, \
                'Network should predict a 2-tuple: hue-map and chroma-map.'
            hue_map = outputs[0].data
            chroma_map = outputs[1].data
            assert hue_map.size() == chroma_map.size(), \
                'Outputs should have same dimensions.'
            sz_h = hue_map.size()
            sz_im = img.size()
            assert sz_im[2]==sz_h[2] and sz_im[3]==sz_h[3], \
                'Spatial dims should match for input and output.'
        elif val_loader.dataset.bins == 'soft':
            from torch.autograd import Variable
            inputs = Variable(img)
            if cuda:
            	inputs = inputs.cuda()
            outputs = model(inputs)
            # TODO: assertions
            # del inputs, outputs
        import pdb; pdb.set_trace()  # breakpoint 0632fd52 //

        model.train()    
    else:
        pass


    # -----------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------
    trainer = train.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()



def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash


def get_log_dir(model_name, config_id, cfg, verbose=True):
    # load config
    name = 'MODEL-%s_CFG-%03d' % (model_name, config_id)
    if verbose:
        for k, v in cfg.items():
            v = str(v)
            if '/' in v:
                continue
            name += '_%s-%s' % (k.upper(), v)
    now = datetime.datetime.now(pytz.timezone('US/Eastern'))
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    log_dir = osp.join(here, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


if __name__ == '__main__':
    main()
