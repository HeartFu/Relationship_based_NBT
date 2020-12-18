import json
import os
import time
import torch
import base64
import numpy as np

from torch.utils import data


class BoxesDataset(data.Dataset):
    def __init__(self, info_path, union_feat_path, boxes_feat_path):
        print('loading information file into memory...')
        tic = time.time()
        self.union_info = json.load(open(info_path, 'r'))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        print('loading union boxes features file into memory...')
        tic = time.time()
        self.union_feat_dict = json.load(open(union_feat_path, 'r'))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        print('loading objects boxes features file into memory...')
        tic = time.time()
        self.boxes_feat_dict = json.load(open(boxes_feat_path, 'r'))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def __len__(self):
        return len(self.union_info)

    def __getitem__(self, index):
        union_boxes = self.union_info[index]
        obj_id = union_boxes['object']
        sub_id = union_boxes['subject']
        union_id = union_boxes['union_boxes_id']
        y = union_boxes['predicate']

        obj_feat = torch.from_numpy(np.frombuffer(base64.b64decode(self.boxes_feat_dict[str(obj_id)]), dtype=np.float32))
        sub_feat = torch.from_numpy(np.frombuffer(base64.b64decode(self.boxes_feat_dict[str(sub_id)]), dtype=np.float32))
        union_feat = torch.from_numpy(np.frombuffer(base64.b64decode(self.union_feat_dict[str(union_id)]), dtype=np.float32))

        return obj_feat, sub_feat, union_feat, y

# class BoxesDataset(data.Dataset):
#     def __init__(self, json_path, boxes_path, union_path):
#         print('loading annotations into memory...')
#         tic = time.time()
#         dataset_json = json.load(open(json_path, 'r'))
#         self.dataset = dataset_json
#         self.boxes_path = boxes_path
#         # self.boxes_feature = self.load_feature(boxes_path)
#         self.union_path = union_path
#         # self.union_feature = self.load_feature(union_path)
#         print('Done (t={:0.2f}s)'.format(time.time() - tic))
#
#     def __getitem__(self, index):
#
#         data = self.dataset[index]
#         obj_id = data['object']
#         sub_id = data['subject']
#         union_id = data['union_boxes_id']
#         y = data['predicate']
#
#         obj_path = os.path.join(self.boxes_path, str(obj_id) + '.txt')
#         obj_feat_file = open(obj_path)
#         obj_feat_code = obj_feat_file.readlines()[0]
#         obj_feat = torch.from_numpy(np.frombuffer(base64.b64decode(obj_feat_code), dtype=np.float32))
#
#         sub_path = os.path.join(self.boxes_path, str(sub_id) + '.txt')
#         sub_feat_file = open(sub_path)
#         sub_feat_code = sub_feat_file.readlines()[0]
#         sub_feat = torch.from_numpy(np.frombuffer(base64.b64decode(sub_feat_code), dtype=np.float32))
#
#         union_path = os.path.join(self.union_path, str(union_id) + '.txt')
#         union_feat_file = open(union_path)
#         union_feat_code = union_feat_file.readlines()[0]
#         union_feat = torch.from_numpy(np.frombuffer(base64.b64decode(union_feat_code), dtype=np.float32))
#
#         return obj_feat, sub_feat, union_feat, y
#
#     def load_feature(self, path):
#         feature_dict = dict()
#         list_file = os.listdir(path)
#         for file in list_file:
#             feature = self.open_feature(path + file)
#             feature_dict[file.replace('.txt', '')] = feature
#         return feature_dict
#
#     def open_feature(self, path):
#         f = open(path)
#         lines = f.readlines()
#         feature = lines[0]
#         f.close()
#         return feature
#
#     def __len__(self):
#         return len(self.dataset)
