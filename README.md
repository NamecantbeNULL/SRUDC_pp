# Modeling Scattering Effect for Under-Display Camera Image Restoration


> **Abstract:** 
The under-display camera (UDC) provides consumers with a full-screen visual experience without any obstruction due to notches or punched holes. However, the semi-transparent nature of the display inevitably introduces severe degradation into UDC images. In this work, we address the UDC image restoration problem with the specific consideration of the scattering effect caused by the display. We explicitly model the scattering effect by treating the display as a piece of homogeneous scattering medium. With the physical model of the scattering effect, we improve the image formation pipeline for the image synthesis to construct a more realistic UDC dataset with ground-truth images. To suppress the scattering effect for the eventual UDC image recovery, a two-branch restoration network is designed. More specifically, the scattering branch leverages the channel-wise self-attention to estimate the parameters of the scattering effect. With the guidance from the scattering branch, the image branch utilizes the local representation advantage of CNN to recover degraded UDC images. A novel channel-wise cross-attention fusion block is devised to integrate the global scattering information into the image branch for the UDC image restoration. To narrow the dark channel distribution gap between the restored and ground-truth images, we specially design a dark channel regularization loss in the training phase. Extensive experiments are conducted on both synthesized and real-world data, demonstrating the superiority of the proposed method over the state-of-the-art UDC restoration techniques.
## Getting started

### Install

We test the code on PyTorch 1.12.1 + CUDA 11.3 + cuDNN 8.3.2.

1. Create a new conda environment
```
conda create -n pt1121 python=3.9
conda activate pt1121
```

2. Install dependencies
```
conda install pytorch=1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Download
### Prepare Dataset

Download and unzip the [Dataset](https://drive.google.com/drive/folders/14Zp2Ff4Ke5491qmyb-BXIx77l9jNMxkG?usp=sharing), and then copy them to `data`.


The final file path should be the same as the following:

```
┬─ save_models
│   ├─ UDC_syn
│   │    ├─ SRUDC-f.pth
│   │    └─ SRUDC-l.pth
│   └─ UDC_SIT
│        ├─ SRUDC-f.pth
│        └─ SRUDC-l.pth
└─ data
    ├─ UDC_syn
    │   ├─ train
    │   │   ├─ HQ_syn
    │   │   │   └─ ... (image filename)
    │   │   └─ LQ_syn
    │   │       └─ ... (corresponds to the former)
    │   └─ test
    │       ├─ HQ_syn
    │       │   └─ ... (image filename)
    │       └─ LQ_syn
    │           └─ ... (corresponds to the former)
    └─ UDC_SIT
```

## Evaluation

### Test

Run the following script to test the trained model:

```sh
python test.py
```

## Citation

If you find this work useful for your research, please cite our paper:


## Acknowledgement

Our code is based on [gUnet](https://github.com/IDKiro/gUNet). We thank the authors for sharing the codes.
