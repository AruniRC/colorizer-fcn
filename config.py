


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    0: dict(
        max_iteration=100000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
    ),

    1: dict(
        max_iteration=500000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=1000,
    ),

    2: dict(
        max_iteration=500000,
        lr=1.0e-12,    # lower learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=1000,
    ),

    3: dict(
        max_iteration=500000,
        lr=1.0e-10,    # lower learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=1000,
    ),

    # debugging on tiny dataset of 9 images
    4: dict(
        max_iteration=500000,
        lr=1.0e-10,    # lower learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=5,
    ),

    # debugging on tiny dataset of 9 images
    # DEBUG: use MSE loss
    5: dict(
        max_iteration=10000,
        lr=1.0e-2,     # changed learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=5,
    ),

    6: dict(
        max_iteration=10000,
        lr=1.0e-10,     # changed learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=10,
    ),

    7: dict(
        max_iteration=10000,
        lr=1.0e-8,     # changed learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=50,
    ),

    8: dict(
        max_iteration=10000,
        lr=1.0e-10,     # changed learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=50,
    ),

    9: dict(
        max_iteration=1000,
        lr=1.0e-4,
        momentum=0.9,  
        weight_decay=0.0005,
        interval_validate=50,
        optim='Adam',
    ),

    # learning rates for Adam optimiser
    10: dict(
        max_iteration=100,
        lr=1.0e-4,
        momentum=0.9,  
        weight_decay=0.0005,
        interval_validate=10,
        optim='Adam',
    ),

    # Adam + training on 10k data
    11: dict(
        max_iteration=100000,
        lr=1.0e-4, # changed learning rate
        momentum=0.9,  
        weight_decay=0.0005,
        interval_validate=50,
        optim='Adam',
    ),

    12: dict(
        max_iteration=100000,
        lr=1.0e-5, # changed learning rate
        momentum=0.8,  
        weight_decay=0.0005,
        interval_validate=50,
        optim='Adam',
    ),

    13: dict(
        max_iteration=100000,
        lr=1.0e-6, # changed learning rate
        momentum=0.8,  
        weight_decay=0.0005,
        interval_validate=50,
        optim='Adam',
    ),

    # low-pass on target image to make task easier, FCN 32s
    14: dict(
        max_iteration=100000,
        lr=1.0e-6, # changed learning rate
        momentum=0.8,  
        weight_decay=0.0005,
        interval_validate=50,
        optim='Adam',
        img_lowpass=8,
        train_set='full',
        val_set='small',
    ),

    # FCN 16s from FCN 32s trained using cfg 14
    15: dict(
        max_iteration=100000,
        lr=1.0e-6, # changed learning rate
        momentum=0.8,  
        weight_decay=0.0005,
        interval_validate=50,
        optim='Adam',
        img_lowpass=8,
        train_set='full',
        val_set='small',
        fcn32s_pretrained_model='/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-014_VCS-db517d6_TIME-20171230-212406/model_best.pth.tar',
    ),

    # FCN 16s - Phase 2, starting from cfg 16 
    16: dict(
        max_iteration=100000,
        lr=1.0e-9, # changed learning rate
        momentum=0.8,  
        weight_decay=0.0005,
        interval_validate=50,
        optim='Adam',
        img_lowpass=8,
        train_set='full',
        val_set='small',
        fcn32s_pretrained_model='/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-014_VCS-db517d6_TIME-20171230-212406/model_best.pth.tar',
        gmm_path='/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn16s_color_CFG-015_VCS-8b659a0_TIME-20180101-055032/gmm.pkl',
        mean_l_path='/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn16s_color_CFG-015_VCS-8b659a0_TIME-20180101-055032/mean_l.npy',
    ),

    17: dict(
        max_iteration=100000,
        lr=1.0e-7, # changed learning rate
        momentum=0.8,  
        weight_decay=0.0005,
        interval_validate=50,
        optim='Adam',
        img_lowpass=8,
        train_set='full',
        val_set='small',
        fcn32s_pretrained_model='/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-014_VCS-db517d6_TIME-20171230-212406/model_best.pth.tar',
        gmm_path='/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn16s_color_CFG-015_VCS-8b659a0_TIME-20180101-055032/gmm.pkl',
        mean_l_path='/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn16s_color_CFG-015_VCS-8b659a0_TIME-20180101-055032/mean_l.npy',
    ),

    # train FCN 8s
    18: dict(
        max_iteration=100000,
        lr=1.0e-6, # changed learning rate
        momentum=0.8,  
        weight_decay=0.0005,
        interval_validate=50,
        optim='Adam',
        img_lowpass=8,
        train_set='full',
        val_set='small',
        fcn16s_pretrained_model='/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn16s_color_CFG-016_VCS-8b659a0_TIME-20180102-211836/model_best.pth.tar',
        gmm_path='/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn16s_color_CFG-015_VCS-8b659a0_TIME-20180101-055032/gmm.pkl',
        mean_l_path='/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn16s_color_CFG-015_VCS-8b659a0_TIME-20180101-055032/mean_l.npy',
    ),

    # train FCN 8s on full-res targets
    # python train_color_fcn8s.py -g 0 -c 19 -b soft -k 16 -m .../MODEL-fcn8s_color_CFG-018.../model_best.pth.tar
    19: dict(
        max_iteration=100000,
        lr=1.0e-6, # changed learning rate
        momentum=0.8,  
        weight_decay=0.0005,
        interval_validate=50,
        optim='Adam',
        img_lowpass=4,
        train_set='full',
        val_set='small',
    ),

    20: dict(
        max_iteration=1.0e+6,
        lr=1.0e-5, # changed learning rate
        momentum=0.9,  
        weight_decay=0.0005,
        interval_validate=50,
        optim='Adam',
        img_lowpass=4,
        train_set='full',
        val_set='full',
        fcn16s_pretrained_model='/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn16s_color_CFG-016_VCS-8b659a0_TIME-20180102-211836/model_best.pth.tar',
    ),

    # ---------------------------------------------------------------------------------------
    # train FCN32s, initialized with pre-trained vgg16
    21: dict(
        max_iteration=3e+5,
        lr=1.0e-7, # changed learning rate
        momentum=0.9,  
        weight_decay=0.0005,
        batch_size=8,
        interval_validate=50,
        optim='Adam',
        img_lowpass=None,
        train_set='full',
        val_set='small',
        gmm_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-021_TIME-20180111-140735/gmm.pkl',
        mean_l_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-021_TIME-20180111-140735/mean_l.npy',
    ),

    22: dict(
        max_iteration=3e+5,
        lr=1.0e-8, # changed learning rate
        momentum=0.9,  
        weight_decay=0.0005,
        batch_size=8,
        interval_validate=50,
        optim='Adam',
        img_lowpass=None,
        train_set='full',
        val_set='small',
        gmm_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-021_TIME-20180111-140735/gmm.pkl',
        mean_l_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-021_TIME-20180111-140735/mean_l.npy',
    ),

    23: dict(
        max_iteration=3e+5,
        lr=1.0e-9, # changed learning rate
        momentum=0.9,  
        weight_decay=0.0005,
        batch_size=8,
        interval_validate=50,
        optim='Adam',
        img_lowpass=None,
        train_set='full',
        val_set='small',
        gmm_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-021_TIME-20180111-140735/gmm.pkl',
        mean_l_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-021_TIME-20180111-140735/mean_l.npy',
    ),

    24: dict(
        max_iteration=3e+5,
        lr=1.0e-10, # changed learning rate
        momentum=0.9,  
        weight_decay=0.0005,
        batch_size=8,
        interval_validate=50,
        optim='Adam',
        img_lowpass=None,
        train_set='full',
        val_set='small',
        gmm_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-021_TIME-20180111-140735/gmm.pkl',
        mean_l_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-021_TIME-20180111-140735/mean_l.npy',
    ),

    # ---------------------------------------------------------------------------------------
    # FCN32s, full ImageNet, 128 color bins - RUNNING
    25: dict(
        max_iteration=1e+5,
        lr=1.0e-6,  
        momentum=0.9,          # not used (Adam)
        weight_decay=0.0005,   
        interval_validate=100,
        optim='Adam',
        img_lowpass=4,
        im_size=(256,256),
        train_set='full',
        val_set='small',
        batch_size=4,
    ),

    # # validated - doesn't work!!!
    # 26: dict(
    #     max_iteration=3e+5,
    #     lr=1.0e-5,  
    #     momentum=0.9,          # not used (Adam)
    #     weight_decay=0.0005,   # not used (Adam)
    #     interval_validate=50,
    #     optim='Adam',
    #     img_lowpass=4,
    #     im_size=(256,256),
    #     train_set='full',
    #     val_set='small',
    #     gmm_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-025_TIME-20180112-153346/gmm.pkl',
    #     mean_l_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-025_TIME-20180112-153346//mean_l.npy',
    # ),

    # RUNNING GMM, curriculum learning - brighter colors stage-1
    27: dict(
        max_iteration=1e+5,
        lr=1.0e-6,  
        momentum=0.9,          # not used (Adam)
        weight_decay=0.0005,   
        interval_validate=100,
        optim='Adam',
        img_lowpass=4,
        im_size=(256,256),
        train_set='bright-1',
        val_set='small',
        batch_size=4,
        gmm_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-027_TIME-20180114-190049/gmm.pkl',
        mean_l_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-027_TIME-20180114-190049/mean_l.npy',
    ),

    # -------------------------------------------------------------------------------------------
    # uniform bins,default sigma + brighter colors stage-1 (32s)
    28: dict(
        max_iteration=1e+5,
        lr=1.0e-6,  
        momentum=0.9,          # not used (Adam)
        weight_decay=0.0005,   
        interval_validate=100,
        optim='Adam',
        img_lowpass=4,
        im_size=(256,256),
        train_set='bright-1',
        val_set='small',
        batch_size=3,
        binning='uniform',
        mean_l_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-028_TIME-20180118-180822/mean_l.npy'
    ),

    # uniform bins,default sigma + brighter colors stage-2 (16s)
    29: dict(
        max_iteration=1e+5,
        lr=1.0e-6,  
        momentum=0.9,          # not used (Adam)
        weight_decay=0.0005,   
        interval_validate=100,
        optim='Adam',
        img_lowpass=4,
        im_size=(256,256),
        train_set='bright-1',
        val_set='small',
        batch_size=4,
        binning='uniform',
        mean_l_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-028_TIME-20180118-180822/mean_l.npy',
        fcn32s_pretrained_model='/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-028_TIME-20180121-211744/model_best.pth.tar',
    ),

    30: dict(
        max_iteration=1e+5,
        lr=1.0e-6,  
        momentum=0.9,          # not used (Adam)
        weight_decay=0.0005,   
        interval_validate=100,
        optim='Adam',
        img_lowpass=4,
        im_size=(256,256),
        train_set='bright-1',
        val_set='small',
        batch_size=4,
        binning='uniform',
        mean_l_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-028_TIME-20180118-180822/mean_l.npy',
        fcn16s_pretrained_model='/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn16s_color_CFG-029_TIME-20180122-151135/model_best.pth.tar',
    ),

    # XXXXXX Loss fluctuating!
    # python -W ignore train_color_fcn8s.py -c 31 -k 256 -b uniform -m <best-model-from-prev-config>
    31: dict(
        max_iteration=1e+5,
        lr=1.0e-7,        # CHANGE: lower the learning  rate
        momentum=0.9,          # not used (Adam)
        weight_decay=0.0005,   
        interval_validate=100,
        optim='Adam',
        img_lowpass=4,      # keep same spatial blurring
        im_size=(256,256),
        train_set='bright-1',
        val_set='small',
        batch_size=4,
        binning='uniform',
        mean_l_path='/home/erdos/arunirc/data1/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn32s_color_CFG-028_TIME-20180118-180822/mean_l.npy',
        fcn16s_pretrained_model='/srv/data1/arunirc/Research/colorize-fcn/colorizer-fcn/logs/MODEL-fcn16s_color_CFG-029_TIME-20180122-151135/model_best.pth.tar',
    ),
    

}



