## HexRUNet PyTorch
An unofficial PyTorch implementation of ICCV 2019 paper ["Orientation-Aware Semantic Segmentation on Icosahedron Spheres"](http://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Orientation-Aware_Semantic_Segmentation_on_Icosahedron_Spheres_ICCV_2019_paper.html). Only HexRUNet-C for Omni-MNIST is implemented right now.

## Requirements
Python 3.6 or later is required.

Python libraries:
- PyTorch >= 1.3.1
- torchvision
- tensorboard
- tqdm
- [igl](https://libigl.github.io/libigl-python-bindings/)


## Training
Run the following command to train with random-rotated training data and evaluate with random-rotated test data.
```bash
python train.py --train_rot --test_rot
```
You can change parameters by arguments (`-h` option for details). 

## Results
Here is the results of this repository. Accuracy of the last epoch (30th epoch) is reported. 

Omni-MNIST HexRUNet-C accuracy (%)
|| N/N | N/R | R/R |
----|----|----|---- 
|This repository | 99.15 | 69.62 | 98.36
|Paper| 99.45 | 29.84 | 97.05

- `N/N`: Non-rotated training and test data
- `N/R`: Non-rotated training data and random-rotated test data
- `R/R`: Random-rotated training and test data

As can be observed here, `N/R` of this repogitory is much higher than the one reported in original paper. I guess it's because the implementation of projecting images on a sphere and rotation are different (My implementation of the projection is based on [ChiWeiHsiao/SphereNet-pytorch](https://github.com/ChiWeiHsiao/SphereNet-pytorch)). 
