## HexRUNet PyTorch
An unofficial PyTorch implementation of ICCV 2019 paper ["Orientation-Aware Semantic Segmentation on Icosahedron Spheres"](http://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Orientation-Aware_Semantic_Segmentation_on_Icosahedron_Spheres_ICCV_2019_paper.html). Only HexRUNet-C for Omni-MNIST is implemented.

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
