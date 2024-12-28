# HNP-FKGC

pytorch==1.11
tqdm==4.64
normflows==1.4
dgl==0.9.0
tensorboardx==2.5.1


## Environment
* python 3.8
* Ubuntu 22.04
* RTX3090
* Memory 32G

## Dataset & Checkpoint
### Original Dataset
* [NELL](https://github.com/xwhan/One-shot-Relational-Learning)
* [FB15K-237](https://github.com/SongW-SW/REFORM)

Download the datasets and extract to the project root folder.  

## Train
NELL 
```bash
python main.py --dataset NELL-One --data_path ./NELL --few 5 --data_form Pre-Train --nellone_5shot_intrain --device 0 --batch_size 128 --flow Planar --g_batch 1024
```
            

FB15K-237 
```bash
python main.py --dataset FB15K-One --data_path ./FB15K --few 5 --data_form Pre-Train --fb15k_5shot_intrain --device 0 --batch_size 128 --flow Planar --g_batch 1024 --eval_batch_size 128 --K 14
```

## Eval
Download the checkpoint and extract to the `state/` folder.

NELL
```bash
python main.py --dataset NELL-One --data_path ./NELL --few 5 --data_form Pre-Train --nellone_5shot_intrain --device 0 --batch_size 128  --step test
```


FB15K-237
```bash
python main.py --dataset FB15K-One --data_path ./FB15K --few 5 --data_form Pre-Train --fb15k_5shot_intrain --device 0 --batch_size 128  --step test
```

