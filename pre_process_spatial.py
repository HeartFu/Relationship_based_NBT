import argparse
import copy
import json

import h5py
from pycocotools.coco import COCO

from misc.dataloader_hdf import HDFSingleDataset
from relation_utils import build_spatial_graph
from tqdm import tqdm


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proposal_h5', type=str, default='data/coco/coco_detection.h5',
                        help='path to the json containing the detection result.')
    parser.add_argument('--ann_file', type=str, default='data/coco/annotations/',
                        help='coco annotation file path')
    parser.add_argument('--input_dic', type=str, default='data/coco/dic_coco.json',
                        help='path to the json containing the preprocessed dataset')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_opt()
    dataloader_ppls = HDFSingleDataset(opt.proposal_h5)

    det_train_path = '%s/instances_train2014.json' % opt.ann_file
    det_val_path = '%s/instances_val2014.json' % opt.ann_file

    coco_train = COCO(det_train_path)
    coco_val = COCO(det_val_path)

    print('DataLoader loading json file: ', opt.input_dic)
    info = json.load(open(opt.input_dic))

    json_info = {}
    # f = h5py.File('data/coco/relationship/spatial_info.h5', 'w')
    for ix in range(len(info['images'])):

        img = info['images'][ix]
        img_id = img['id']
        if img_id != 262284:
            continue
        # img_id = 262284
        file_path = img['file_path']
        coco_split = file_path.split('/')[0]
        # get the ground truth bounding box.
        if coco_split == 'train2014':
            coco = coco_train
        else:
            coco = coco_val

        img_info = coco.imgs[img_id]
        w = img_info['width']
        h = img_info['height']

        proposal_item = copy.deepcopy(dataloader_ppls[ix])
        num_proposal = int(proposal_item['dets_num'])
        num_nms = int(proposal_item['nms_num'])
        proposals = proposal_item['dets_labels']
        proposals = proposals.squeeze()[:num_nms, :]
        bboxes = proposals[:, :4]

        spa_matrix = build_spatial_graph(bboxes, h, w)

        # f[img_id] = spa_matrix
        json_info[img_id] = spa_matrix.tolist()

        # if ix > 50:
        #     break

    # save all boxes

    # f['data'] = json_info
    # f.close()
    b = json.dumps(json_info)
    f2 = open('data/coco/relationship/spatial_info_test.json', 'w')
    f2.write(b)
    f2.close()
