# **Enhancement of Urban Floodwater Mapping From Aerial Imagery With Dense Shadows via Semisupervised Learning**

This is an official PyTorch implementation of a semi-supervised learning framework for flood mapping. The manuscript can be visited via https://ieeexplore.ieee.org/abstract/document/9924583/. The Calgary-Flood datasets used in this paper can be accessed from [[GoogleDirve]](https://drive.google.com/drive/folders/1pNIfaiHdzeL5-hA0sp8ms4wJTCBiRgLz?usp=sharing) or [[BaiduDisk]](https://pan.baidu.com/s/1GfWQIq3J_XVd0MWLhdKVOA?pwd=r5wa).

## 1. Directory Structure    
After obtain the Calgary-Flood datasets, you need to process first and generate lists of image/label files and place as the structure shown below. Every txt file contains the full absolute path of the files, each image/label per line. Note: for `train_unsup_image.txt`, you can just copy `test_image.txt` and then rename it to `train_unsup_image.txt`.
```
/root
    /train_image.txt
    /train_label.txt
    /test_image.txt
    /test_label.txt
    /val_image.txt
    /val_label.txt
    /train_unsup_image.txt
```
## 2. Usage
### Installation
The code is developed using Python 3.8 with PyTorch 1.9.1. The code is developed and tested using singel RTX 2080 Ti GPU.

**(1) Clone this repo.**
```
git clone https://github.com/YJ-He/Flood_Mapping_SSL.git
```

**(2) Create a conda environment.**  
```
conda env create -f environment.yaml
conda activate flood_mapping
```

### Training
1. set `root_dir` and hyper-parameters configuration in `./configs/config.cfg`.
2. run `python train.py`.

### Evaludation
1. set `root_dir` and hyper-parameters configuration in `./configs/config.cfg`.
2. set `pathCkpt` in `test.py` to indicate the model checkpoint file.
3. run `python test.py`.


## 3.Citation
If this repo is useful in your research, please kindly consider citing our paper as follow.
```
@article{he2022enhancement,
  title={Enhancement of Urban Floodwater Mapping From Aerial Imagery With Dense Shadows via Semi-Supervised Learning},
  author={He, Yongjun and Wang, Jinfei and Zhang, Ying and Liao, Chunhua},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2022},
  publisher={IEEE}
}
```

**If our work give you some insights and hints, star me please! Thank you~**


