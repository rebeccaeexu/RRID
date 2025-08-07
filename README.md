# [ECCV 2024] RRID
### Image Demoireing in RAW and sRGB Domains  [#Paper Link](https://arxiv.org/abs/2312.09063)

Shuning Xu, Binbin Song, Xiangyu Chen, Xina Liu and Jiantao Zhou



## Updates

- ✅ 2024-03-15: Release the first version of the paper at Arxiv.
- ✅ 2024-07-01: Release the codes of RRID.
- ✅ 2024-07-11: Release the models and results of RRID.



## Overview

![image-20240701125450731](https://github.com/rebeccaeexu/RRID/blob/main/figs/image-20240701125450731.png)

![image-20240701125517073](https://github.com/rebeccaeexu/RRID/blob/main/figs/image-20240701125517073.png)

> Moiré patterns frequently appear when capturing screens with smartphones or cameras, potentially compromising image quality. Previous studies suggest that moiré pattern elimination in the RAW domain offers greater effectiveness compared to demoiréing in the sRGB domain. Nevertheless, relying solely on RAW data for image demoiréing is insufficient in mitigating the color cast due to the absence of essential information required for the color correction by the image signal processor (ISP). In this paper, we propose to jointly utilize both RAW and sRGB data for image demoiréing (RRID), which are readily accessible in modern smartphones and DSLR cameras. We develop Skip-Connection-based Demoiréing Module (SCDM) with Gated Feedback Module (GFM) and Frequency Selection Module (FSM) embedded in skip-connections for the efficient and effective demoiréing of RAW and sRGB features, respectively. Subsequently, we design a RGB Guided ISP (RGISP) to learn a device-dependent ISP, assisting the process of color recovery. Extensive experiments demonstrate that our RRID outperforms state-of-the-art approaches, in terms of the performance in moiré pattern removal and color cast correction by 0.62dB in PSNR and 0.003 in SSIM.



## Demoireing Results

![image-20240701125619798](https://github.com/rebeccaeexu/RRID/blob/main/figs/image-20240701125619798.png)

![image-20240701125646976](https://github.com/rebeccaeexu/RRID/blob/main/figs/image-20240701125646976.png)

**Fig. 3: Qualitative comparison on RAW image demoiréing TMM22 dataset**



## Environment

- basicsr==1.4.2
- scikit-image==0.15.0
- deepspeed



## Prepare

1. Download [TMM22 dataset](https://pan.baidu.com/s/1RqQHV4FO49wPID5-vtoRaQ?pwd=3c6m).
2. Download the [pre-trained model](https://www.dropbox.com/scl/fo/wxhxlj6y064fbx4lrotcd/APoEwRwRT82LnW8wzDqvW24?rlkey=ghf505sfpkr8y9z5psourzpk6&st=enbcm5va&dl=0).



## How To Test

```
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/Test.yml
```



## How To Train

* Single GPU training

```python
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/Train.yml
```

* Distributed training

```python
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train/Train.yml --launcher pytorch
```



## Results

The inference results on benchmark datasets are available at [Dropbox link](https://www.dropbox.com/scl/fo/1a0mhgy6x76zi2bww7fs1/AB7rQtX2bdvJxBEtv4e54yM?rlkey=esb6931y40s9vjsqu4q740ckt&st=kqpualkl&dl=0).



## Citations

#### BibTeX

    @article{xu2023image,
      title={Image Demoireing in RAW and sRGB Domains},
      author={Xu, Shuning and Song, Binbin and Chen, Xiangyu and Zhou, Jiantao},
      journal={arXiv preprint arXiv:2312.09063},
      year={2023}
    }

