# C3R_UDA

Implementation for the paper entitled "C$^3$R: Category Contrastive Adaptation and Consistency Regularization for Cross-Modality Medical Image Segmentation"

## Table of Contents

- [C3R\_UDA](#c3r_uda)
  - [Table of Contents](#table-of-contents)
  - [Environment requirements](#environment-requirements)
  - [Dataset](#dataset)
  - [Project Tree](#project-tree)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Environment requirements

We test the code on 1 single V100-32GB GPU, or 4 TITAN-XP-12GB GPU. 

Note that the code is tested on python 3.7.10, and the cuda version is 10.2.

```
imgaug==0.4.0
matplotlib==3.4.2
MedPy==0.3.0
numpy==1.20.3
Pillow==8.4.0
PyYAML==6.0
scikit_learn==1.0.2
tensorboardX==2.4.1
torch==1.3.1
torchsummary==1.5.1
torchvision==0.4.2
tqdm==4.46.0
```

## Dataset

Some steps to prepare the dataset:

1. Download the preprocessed data from [SIFA](https://github.com/cchen-cc/SIFA).
2. Decode the `tfrecords` data to `npz` data using the preprocessing code as:

```
    python preprocess.py --data_dir./data --save_dir./data
```

3. Please move the decoded data to the `data` folder.



## Project Tree
```
C3R_UDA
├─ configs                # hyperparameters
│  └─ config_mr2ct.yml    
│  └─ config_ct2mr.yml    
├─ data                   # mmwhs dataset or other datasets
│  ├─ training_mr
│  │  ├─ xxxx.npz
│  │  └─ xxxx.npz
│  ├─ training_ct
│  │  ├─ xxxx.npz
│  │  └─ xxxx.npz
│  ├─ validation_mr
│  │  ├─ xxxx.npz
│  │  └─ xxxx.npz
│  ├─ validation_ct
│  │  ├─ xxxx.npz
│  │  └─ xxxx.npz
│  ├─ test_mr
│  │  ├─ xxxx.npz
│  │  └─ xxxx.npz
│  └─ test_ct
│  │  ├─ xxxx.npz
│  │  └─ xxxx.npz
├─ datasets               # data loader
│  ├─ __init__.py
│  ├─ augmentations.py
│  ├─ ct_dataset.py
│  └─ mr_dataset.py
├─ losses
│  ├─ __init__.py         
│  ├─ gan_loss.py
│  └─ seg_loss.py
├─ models                 # model architecture and training details
│  ├─ __init__.py
│  ├─ c3r_trainer.py
│  ├─ discriminator.py
│  ├─ res_parts.py
│  ├─ seg_model.py
│  └─ unet_parts.py
├─ utils
│  ├─ __init__.py
│  ├─ utils.py
│  └─ visualizer.py
├─ train.py               # training entry
└─ evaluate.py            # evaluation entry
```

## Training

```
python train.py --config configs/config_mr2ct.yml  # for mr2ct
python train.py --config configs/config_ct2mr.yml  # for ct2mr
```

## Evaluation

```
python run_eval.py --target ct  --model_path pretrained_mr2ct.pt  # for mr2ct
python run_eval.py --target mr  --model_path pretrained_ct2mr.pt  # for ct2mr
```

## Citation

If you find the code useful for your research, please cite:
```
To be updated.
```

## Acknowledgement

* We thank the repo of SIFA, you can find the preprocessed data in [SIFA](https://github.com/cchen-cc/SIFA).
* The style of the project organization is borrowed from [CAG_UDA](https://github.com/RogerZhangzz/CAG_UDA).