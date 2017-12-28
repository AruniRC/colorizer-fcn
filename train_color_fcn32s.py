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


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash


def get_log_dir(model_name, config_id, cfg):
    # load config
    name = 'MODEL-%s_CFG-%03d' % (model_name, config_id)
    for k, v in cfg.items():
        v = str(v)
        if '/' in v:
            continue
        name += '_%s-%s' % (k.upper(), v)
    now = datetime.datetime.now(pytz.timezone('US/Eastern'))
    name += '_VCS-%s' % git_hash()
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    # TODO - create out at custom location
    log_dir = osp.join(here, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        models.FCN32sColor
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, nn.BatchNorm2d):
            # TODO - check with BN examples if correct procedure!
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))


here = osp.dirname(osp.abspath(__file__))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('-b', '--binning', default='one-hot', 
                        choices=('soft','one-hot'))
    parser.add_argument('-k', '--numbins', type=int, default=32)
    parser.add_argument('-d', '--dataset_path', 
                        default='/data/arunirc/datasets/ImageNet/images')
    parser.add_argument('-m', '--model_path', default=None)
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    cfg = configurations[args.config]
    out = get_log_dir('fcn32s_color', args.config, cfg) # TODO - change dir
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
    
    # DEBUG: set='tiny'
    train_loader = torch.utils.data.DataLoader(
        data_loader.ColorizeImageNet(root, split='train', \
        bins=args.binning, log_dir=out, num_hc_bins=args.numbins, set='tiny'),
        batch_size=1, shuffle=False, **kwargs) # DEBUG: set shuffle False

    # val_loader = torch.utils.data.DataLoader(
    #     torchfcn.datasets.ColorizeImageNet(root, split='val', \
    #     bins=args.binning, log_dir=out, num_hc_bins=args.numbins),
    #     batch_size=1, shuffle=False, **kwargs) 

    # DEBUG: set='tiny'
    val_loader = torch.utils.data.DataLoader(
        data_loader.ColorizeImageNet(root, split='val', \
        bins=args.binning, log_dir=out, num_hc_bins=args.numbins, set='tiny'),
        batch_size=1, shuffle=False, **kwargs)


    # -----------------------------------------------------------------------------
    # 2. model
    # -----------------------------------------------------------------------------
    model = models.FCN32sColor(n_class=args.numbins, bin_type=args.binning)
    if args.model_path:
        checkpoint = torch.load(args.model_path)        
        model.load_state_dict(checkpoint['model_state_dict'])
    # for param in model.parameters():
    #     param.requires_grad = False
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
    optim = torch.optim.SGD(params,
					        lr=cfg['lr'],
					        momentum=cfg['momentum'],
					        weight_decay=cfg['weight_decay'])

    # TODO - choose optimizer - Adam or SGD

    # optim = torch.optim.SGD(
    #     [
    #         {'params': get_parameters(model, bias=False)},
    #         {'params': get_parameters(model, bias=True),
    #          'lr': cfg['lr'] * 2, 'weight_decay': 0},
    #     ],
    #     lr=cfg['lr'],
    #     momentum=cfg['momentum'],
    #     weight_decay=cfg['weight_decay'])

    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])


    # -----------------------------------------------------------------------------
    # Sanity-check: forward pass with a single sample
    # -----------------------------------------------------------------------------
    # dataiter = iter(val_loader)
    # img, label = dataiter.next()
    # model.eval()
    # if val_loader.dataset.bins == 'one-hot':
    #     from torch.autograd import Variable
    #     inputs = Variable(img)
    #     if cuda:
    #         inputs = inputs.cuda()
    #     outputs = model(inputs)
    #     assert len(outputs)==2, \
    #         'Network should predict a 2-tuple: hue-map and chroma-map.'
    #     hue_map = outputs[0].data
    #     chroma_map = outputs[1].data
    #     assert hue_map.size() == chroma_map.size(), \
    #         'Outputs should have same dimensions.'
    #     sz_h = hue_map.size()
    #     sz_im = img.size()
    #     assert sz_im[2]==sz_h[2] and sz_im[3]==sz_h[3], \
    #         'Spatial dims should match for input and output.'
    # elif val_loader.dataset.bins == 'soft':
    #     from torch.autograd import Variable
    #     inputs = Variable(img)
    #     if cuda:
    #       inputs = inputs.cuda()
    #     outputs = model(inputs)
    #     # TODO: assertions
    #     # del inputs, outputs
    # model.train()



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


if __name__ == '__main__':
    main()
