## Colorize FCN

Learn to colorize grayscale images using the FCN segmentation architecture.

### Setup

Install PyTorch and TorchVision in an Anaconda environment then install the dependencies (using conda) from the `environment.yml` file.

For 1080 GPUs, use: `conda install pytorch torchvision cuda80 -c soumith`.
This addresses the massive slowdown in executing `model.cuda()`.

### Usage

Experiment settings such as learning rate are defined in `config.py`, each setting as a key in a Python dict. Training FCN-32s (stride of 32 px) is done first, using `train_color_fcn32s.py` and specifying a configuration (the input args are explained in the Python script).

### Experiments

* **Tiny dataset** - sanity-checks by over-fitting an FCN-32s model from scratch on a dataset of 9 images. The command to train this network is `python train_color_fcn32s.py -g 0 -c 7 -b soft -k 32`. This trains the network to predict "color labels" (32 GMM clusters in Hue-Chroma space) at each pixel given the single-channel Lightness image as input. The KL-divergence loss (PyTorch implementation) is used between GMM posteriors as targets and the network outputs. The data paths set as defaults  can be changed in the training script. The results are saved under `./logs` with the latest timestamp. **Notes** - BatchNorm, mean-centering input image,  Adam optimizer, fixed bilinear upsampler layer. 

For visualizing the labeled regions in the image, the _average RGB_ within each labeled region is used. 

[Color image | target clusters | grayscale image | predicted clusters ]
![viz results tiny](figures/fcn32s-tiny-iter1000.jpg)

* **FCN 32s simplified training** - Given that the FCN 32s model is by construction not designed for very fine-grained spatial segmentation (due to its large 32 pixel stride), we make the task easier by reducing the number of GMM color clusters and removing high-frequency terms from the target label mask. This makes the task simple enough for the network to train on readily, and give reasonable, although rough, _validation set_ results after 100k iterations. Config 14 in `config.py` is used. 

![viz results fcn32s](figures/fcn32s_14_iter100000.jpg)

![val loss fcn32s](figures/fcn32s_c14_val_loss.png)


