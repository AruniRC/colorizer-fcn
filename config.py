


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

    # low-pass on image to make task easier, FCN 32s
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

    16: dict(
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

}



