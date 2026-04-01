# [ECCV 2024] RRID
### Image Demoireing in RAW and sRGB Domains  [#Paper Link](https://arxiv.org/abs/2312.09063)

Shuning Xu, Binbin Song, Xiangyu Chen, Xina Liu and Jiantao Zhou



## Updates

- ✅ 2024-03-15: Release the first version of the paper at Arxiv.
- ✅ 2024-07-01: Release the codes of RRID.
- ✅ 2024-07-11: Release the models and results of RRID.


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

    @inproceedings{xu2024image,
      title={Image demoireing in raw and srgb domains},
      author={Xu, Shuning and Song, Binbin and Chen, Xiangyu and Liu, Xina and Zhou, Jiantao},
      booktitle={European conference on computer vision},
      pages={108--124},
      year={2024},
      organization={Springer}
    }
    
