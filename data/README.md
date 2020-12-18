## Data Preparation for Relationship-based Neural Baby Talk
### Image Dataset

- COCO: Download coco images from [link](http://cocodataset.org/#download), we need `2014 training` images and `2014 val` images. You should put the image in some directory, denoted as `$IMAGE_ROOT`.

### Pretrained CNN weight
- Download pretrained CNN weight from [link](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth), rename to `resnet101.pth` and put it into `imagenet_weights/`

### COCO

You can either download the all pre-processed data from [here](https://drive.google.com/file/d/1dIJuG78qfxhxHmm3ID--efFgpOFYU-5R/view?usp=sharing) or following the below steps to get the pre-processed data one by one.

- Download the preprocessed Karpathy's split of coco caption from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). Extract `dataset_coco.json` from the zip file and copy it into `coco/`.
- Download COCO 2014 Train/Val annotations from [link](http://images.cocodataset.org/annotations/annotations_trainval2014.zip). Extract the zip file and put the json file under `coco/annotations/`
- Download stanford core nlp tools and modified the `scripts/prepro_dic_coco.py` with correct stanford core nlp location. (In my experiment, I use the the version of `stanford-corenlp-full-2017-06-09` [link](https://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip))
- You can either download the preprocessed data from [here](https://drive.google.com/file/d/1TuW9YwuxjSsyPbUVTqCVr_RfhrGlaDD0/view?usp=sharing) or you can use the pre-process script to generate the data. Under the `root` directory, run the following command to pre-process the data.
```
python prepro/prepro_dic_coco.py --input_json data/coco/dataset_coco.json --split normal --output_dic_json data/coco/dic_coco.json --output_cap_json data/coco/cap_coco.json
```
- Download the pre-extracted coco detection result from [link](https://drive.google.com/file/d/1G9HhJWyyxB9MROZ-qTidYc85hbbILnBR/view?usp=sharing) and copy it into `coco/`.
- Download the pre-extracted relationships results from [link](https://drive.google.com/file/d/1GDJUYNFFJqKXBLMB7bUO4A2C453rgWGX/view?usp=sharing) and extract the files to copy them into `coco/relationship/`.
- After all these steps, we are ready to train the model for coco :)
