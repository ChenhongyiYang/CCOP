# CCOP

------

Code of our paper "Contrastive Object-level Pre-training with Spatial Noise Curriculum Learning"

## Requirement

------

1. Install [OpenSelfSup](https://github.com/open-mmlab/OpenSelfSup) 
2. Install [Detectron2](https://github.com/facebookresearch/detectron2), Do not forget to setup Detectron2 datasets!!!
3. Install [Kornia](https://github.com/kornia/kornia) for fast data augmentation

## Usage

------

### Run Selective Search

```shell
% remember to setup the dataset paths
python tools/selective_search.py
```

### Setup dataset

```shell
mkdir data
ln -s path_to_coco data
```

### Run CCOP pre-training and Mask R-CNN benchmark

```shell
% training a ResNet-50 model with 8 GPU
zsh tools/det_train_benchmark.sh configs/selfsup/ccop/r50_d2.py 8 path_to_output
```

## Citation

------

```
@article{yang2021contrastive,
  title={Contrastive Object-level Pre-training with Spatial Noise Curriculum Learning},
  author={Yang, Chenhongyi and Huang, Lichao and Crowley, Elliot J},
  journal={arXiv preprint arXiv:2111.13651},
  year={2021}
}
```