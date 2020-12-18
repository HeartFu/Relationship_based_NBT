# Relationship-based Neural Baby Talk

This project are run in anaconda virtual environment.


## requirement

Inference:

- [pytorch](http://pytorch.org/)
- [torchvision](https://github.com/pytorch/vision)
- [torchtext](https://github.com/pytorch/text)

Data Preparation:

- [stanford-corenlp-wrapper](https://github.com/Lynten/stanford-corenlp)
- [stanford-corenlp](https://stanfordnlp.github.io/CoreNLP/)



## Training and Evaluation
### Data Preparation
Head to `data/README.md`, and prepare the data for training and evaluation.

### Pretrained model

These model are pre-trained based on Graph Attention Mechanism.

| Model | Dataset | Backend | Batch size | Link  |
| ---- | :----:| :----:| :----:|:----:|
| R-NBT using implicit relationships | COCO | Res-101 | 100 | [Pre-trained Model](https://drive.google.com/file/d/1qd-e198n-G4LSSNDFmpdseAW9PNhp4Ws/view?usp=sharing) |
| R-NBT using spatial relationships | COCO | Res-101 | 100 | [Pre-trained Model](https://drive.google.com/file/d/171-7S95EtscnSTH_8G8uNGKH119hsIw5/view?usp=sharing) |
| R-NBT using semantic relationships | COCO | Res-101 | 100 | [Pre-trained Model](https://drive.google.com/file/d/1tOcqHMowyJUdYJWSKbNYr3M_H9IyQYy8/view?usp=sharing) |

We also provided NBT with graph convolutional network to compare with R-NBT.

| Model | Dataset | Backend | Batch size | Link  |
| ---- | :----:| :----:| :----:|:----:|
| GCN with spatial relationships| COCO | Res-101 | 100 | [Pre-trained Model](https://drive.google.com/file/d/1AsrC9gSnJz3Xd2xhyS6mUoVHR0RPW-DP/view?usp=sharing) |
| GCN with semantic relationships | COCO | Res-101 | 100 | [Pre-trained Model](https://drive.google.com/file/d/1kVF_RIstODNA7GGsSg90v7suqC-rTDKs/view?usp=sharing) |

##### Training (COCO)

Firstly, modify the config file `cfgs/normal_coco_res101.yml` with the correct file path.

```
python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 100 --cuda True --num_workers 10 --max_epoch 50 --relation_type spatial --graph_attention True --mGPUs True
```
##### Evaluation (COCO)
Download all Pre-trained model. Extract the file and put it under `save/`. Modify the weights path in commands below to make sure the correct weights.

```
python evaluation.py --path_opt cfgs/normal_coco_res101.yml --batch_size 100 --cuda True --num_workers 1 --beam_size 3 --sem_model --spa_model --imp_model --sem_pro 0.3 --spa_pro 0.3 --imp_pro 0.4 --sem_start_from save/bs100_coco_sem/ --spa_start_from save/bs100_coco_1024/ --imp_start_from save/bs100_coco_imp/ --graph_attention True
```

If you just want to evaluate model with single relationships, following this command:

```
python evaluation.py --path_opt cfgs/normal_coco_res101.yml --batch_size 100 --cuda True --num_workers 1 --beam_size 3 --sem_model --sem_pro 1 --sem_start_from save/bs100_coco_sem/  --graph_attention True
```

Note that "sem_model" with "sem_pro" and "sem_start_from" can change to other types of relationships, which are "imp_model" and "spa_model" with correct probability and weights path.

For two relationships, follow the command:

```
python evaluation.py --path_opt cfgs/normal_coco_res101.yml --batch_size 100 --cuda True --num_workers 1 --beam_size 3 --sem_model --imp_model --sem_pro 0.5 --imp_pro 0.5 --sem_start_from save/bs100_coco_sem/ --imp_start_from save/bs100_coco_imp/ --graph_attention True
```

Note that "sem_model" and "imp_model" can change to "sem_model" & "spa_model", "imp_model" & "spa_model" with correct probabilities and weights paths.

For greedy search, change beam_size to 1 in above commands.

### Multi-GPU Training
This codebase also support training with multiple GPU. To enable this feature, simply add `--mGPUs Ture` in the commnad.


## Acknowledgement

We thank Jiasen Lu et al. for [Neural Baby Talk](https://github.com/jiasenlu/NeuralBabyTalk) repo.
