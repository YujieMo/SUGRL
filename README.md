# SUGRL: Simple Unsupervised Graph Representation Learning

This repository contains the reference code for the paper Simple Unsupervised Graph Representation Learning 

## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Training](#train)
0. [Testing](#test)

## Installation
pip install -r requirements.txt 

## Preparation

Pretrained model see >>>[here](checkpoints/)<<<.

Configs see >>>[here](args.yaml)<<<.

Dataset (`--dataset-class`, `--dataset-name`,`--Custom-key`)

| Dataset class          | Dataset name | Custom key    |
|------|------|------|
| Planetoid              | Cora         | classification|
| Planetoid              | CiteSeer     | classification|
| Planetoid              | PubMed       | classification|
| MyAmazon               | Photo        | classification|
| MyAmazon               | Computers    | classification|
| PygNodePropPredDataset | ogbn-arxiv   | classification|
| PygNodePropPredDataset | ogbn-mag     | classification|
| PygNodePropPredDataset | ogbn-products| classification|

Important args:
* `--pretrain` Test checkpoints
* `--dataset-class` Planetoid, MyAmazon, PygNodePropPredDataset
* `--dataset-name` Cora, CiteSeer, PubMed, Photo, Computers, ogbn-arxiv, ogbn-mag, ogbn-products
* `--custom_key` classification, link, clu


## Training

```shell
python train.py 
```


## Testing
Choose the custom_key of different downstream tasks

## Citation
```shell
@InProceedings{Mo_AAAI_2022, 
title={Simple Unsupervised Graph Representation Learning}, 
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)}, 
author={Mo, Yujie and Peng, Liang and Xu, Jie and Shi, Xiaoshuang and Zhu, Xiaofeng},
year={2022}, 
pages={7797-7805} 
}
```
