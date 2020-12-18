# Semantic Relationship Classifier

## Training and Evaluation

### Data Processing

- Download the relationship features of VG dataset from [link](https://drive.google.com/file/d/18hxTep08VOk52hXiNGeEMGzyxUMhYceA/view?usp=sharing) which we have already extracted by Detectron2. 

You also can follow the Detectron2 code to extract the features, the repo is [here](https://github.com/HeartFu/py-bottom-up-attention).


### Pretrained model

| Model | Dataset | Batch size | Link  |
| ---- | :----:| :----:| :----:|
| Classifier | VG |  1024 | [Pre-trained Model](https://drive.google.com/file/d/1X3dQaiVaI-aQ5PirKAPA_2doicR04MdJ/view?usp=sharing) |

## Training

```
python trainer.py --data_path vg/data/ --boxes_path vg/data/boxes_path/ --union_path vg/data/union_path/
```

- data_path: The path of relationship information json file.
- boxes_path: The path of bounding boxes features from Detectron2.
- union_path: The path of union bounding boxes features from Detectron2.

## evaluation

```
python evaluation.py --data_path vg/data/ --boxes_path vg/data/boxes_path/ --union_path vg/data/union_path/ --checkpoint checkpoint/weight_norm/
```