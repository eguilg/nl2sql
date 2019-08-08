## [阿里天池首届中文NL2SQL挑战赛](https://tianchi.aliyun.com/competition/entrance/231716/introduction) 
排名: 8

队名: 爆写规则一万行

成员: [eguilg](https://github.com/eguilg), [严之zh](https://github.com/zhangyan333), [naniwet](https://github.com/naniwet)

### Environments
Ubuntu 18.04

Python: 3.6.5

Pytorch: 1.1.0 

CUDA: 9.0

CUDNN: 7.1.3

### Required packages
We used [pytorch-pretrained-bert](https://pypi.org/project/pytorch-pretrained-bert/) package for backbone BERT model. 

(**Note that** the original [pytorch-pretrained-bert](https://pypi.org/project/pytorch-pretrained-bert/) was updated to [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) during the contest, but we chose to stick to the old version for stability.)

Required Python packages:
```
fuzzywuzzy==0.17.0
numpy==1.17.0
torch==1.1.0
pytorch-pretrained-bert==0.6.2
tqdm==4.24.0
records 
```
Command to install the required python packages:
```
pip install -r requirements.txt
```

### [Train/Test](code/README.md)

