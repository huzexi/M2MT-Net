# M2MT-Net
This repository is the official implementation of the paper "Beyond Subspace Isolation: Many-to-Many Transformer for Light Field Image Super-resolution" published in TMM 2024.

[[arxiv](https://arxiv.org/abs/2401.00740)] [[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10812790)]

<img src="https://raw.github.com/huzexi/M2MT-Net/main/figs/M2MT.jpg" width="500">
<img src="https://raw.github.com/huzexi/M2MT-Net/main/figs/Efficiency.jpg" width="1000">

## Setup
The repo is based on [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR) and provides the minimal code to reproduce the results in the paper.

Please refer to the repo for preliminaries such as dataset preparation before testing.

## Testing
Download the pretrained weight files from the repo's release page.
To test the model, run the following command:
```bash
# For 4x LFSR
python test.py --scale_factor 4 --model_name M2MTNet --path_pre_pth weights/M2MTNet.4x.n9.pth --path_for_test data_for_test/
# For 2x LFSR
python test.py --scale_factor 2 --model_name M2MTNet --path_pre_pth weights/M2MTNet.2x.n8.pth --path_for_test data_for_test/
```
Before testing the 2x model, please change the config in `model/SR/M2MTNet/M2MTNetConfig.py` to `n_block=9`.

## Benchmark
### 2x LFSR
|                            Methods                             | Scale |         EPFL          |       HCInew        |       HCIold        |         INRIA         |       STFgantry       |
|:--------------------------------------------------------------:|:-----:|:---------------------:|:-------------------:|:-------------------:|:---------------------:|:---------------------:|
|          [**DPT**](https://github.com/BITszwang/DPT)           |  2x   |     34.490/0.9758     |    37.355/0.9771    |    44.302/0.9943    |     36.409/0.9843     |     39.429/0.9926     |
|        [**LFT**](https://github.com/ZhengyuLiang24/LFT)        |  2x   |     34.804/0.9781     |    37.838/0.9791    |    44.522/0.9945    |     36.594/0.9855     |     40.510/0.9941     | 
|    [**DistgSSR**](https://github.com/YingqianWang/DistgSSR)    |  2x   |     34.809/0.9787     |    37.959/0.9796    |    44.943/0.9949    |     36.586/0.9859     |     40.404/0.9942     |
|   [**LFSSR_SAV**](https://github.com/Joechann0831/SAV_conv)    |  2x   |     34.616/0.9772     |    37.425/0.9776    |    44.216/0.9942    |     36.364/0.9849     |     38.689/0.9914     |
|       [**EPIT**](https://github.com/ZhengyuLiang24/EPIT)       |  2x   |     34.826/0.9775     |    38.228/*0.9810*  |  *45.075*/0.9949    |     36.672/0.9853     | **42.166**/**0.9957** |
|    [**HLFSR-SSR**](https://github.com/duongvinh/HLFSR-SSR)     |  2x   |   *35.310*/*0.9800*   |   *38.317*/0.9807   |    44.978/*0.9950*  |   *37.060*/*0.9867*   |     40.849/0.9947     |
|         [**LF-DET**](https://github.com/Congrx/LF-DET)         |  2x   |     35.262/0.9797     |    38.314/0.9807    |    44.986/*0.9950*  |     36.949/0.9864     |   *41.762*/*0.9955*   |
|     [**M2MT-Net**](https://github.com/huzexi/M2MT-Net)         |  2x   | **35.877**/**0.9822** |**38.476**/**0.9812**|**45.344**/**0.9953**| **37.460**/**0.9872** |     40.987/0.9949     |

### 4x LFSR
|                            Methods                             | Scale |         EPFL          |        HCInew         |        HCIold         |         INRIA         |      STFgantry      |
|:--------------------------------------------------------------:|:-----:|:---------------------:|:---------------------:|:---------------------:|:---------------------:|:-------------------:|
|          [**DPT**](https://github.com/BITszwang/DPT)           |  4x   |     28.939/0.9170     |     31.196/0.9188     |     37.412/0.9721     |     30.964/0.9503     |    31.150/0.9488    |
|        [**LFT**](https://github.com/ZhengyuLiang24/LFT)        |  4x   |     29.255/0.9210     |     31.462/0.9218     |     37.630/0.9735     |     31.205/0.9524     |    31.860/0.9548    |
|    [**DistgSSR**](https://github.com/YingqianWang/DistgSSR)    |  4x   |     28.992/0.9195     |     31.380/0.9217     |     37.563/0.9732     |     30.994/0.9519     |    31.649/0.9535    |
|   [**LFSSR_SAV**](https://github.com/Joechann0831/SAV_conv)    |  4x   |     29.368/0.9223     |     31.450/0.9217     |     37.497/0.9721     |     31.270/0.9531     |    31.362/0.9505    |
|       [**EPIT**](https://github.com/ZhengyuLiang24/EPIT)       |  4x   |     29.339/0.9197     |     31.511/0.9231     |     37.677/0.9737     |     31.372/0.9526     |  *32.179*/0.9571    |
|    [**HLFSR-SSR**](https://github.com/duongvinh/HLFSR-SSR)     |  4x   |     29.196/0.9222     |    *31.571*/*0.9238*  |     37.776/0.9742     |     31.241/*0.9534*   |    31.641/0.9537    |
|         [**LF-DET**](https://github.com/Congrx/LF-DET)         |  4x   |   *29.473*/*0.9230*   |     31.558/0.9235     |    *37.843*/*0.9744*  |   *31.389*/*0.9534*   |    32.139/*0.9573*  |
|      [**M2MT-Net**](https://github.com/huzexi/M2MT-Net)        |  4x   | **29.852**/**0.9284** | **31.761**/**0.9261** | **37.982**/**0.9749** | **31.771**/**0.9563** |**32.205**/**0.9584**|

## Citiation
```
@ARTICLE{M2MT-Net,
  author={Hu, Zeke Zexi and Chen, Xiaoming and Chung, Vera Yuk Ying and Shen, Yiran},
  journal={IEEE Transactions on Multimedia},
  title={Beyond Subspace Isolation: Many-to-Many Transformer for Light Field Image Super-Resolution},
  year={2025},
  volume={27},
  number={},
  pages={1334-1348},
  doi={10.1109/TMM.2024.3521795}
}
```

## Acknowledgement
**We thank [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR) for their foundational contributions that have greatly benefited the light field community.**
