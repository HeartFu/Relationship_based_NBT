import json
import os

from tqdm import tqdm


def load_feature(path, json_path):
    feature_dict = dict()
    json_file = json.load(open(json_path, 'r'))
    for info in tqdm(json_file):
        obj_id = info['object']
        sub_id = info['subject']
        if obj_id not in feature_dict.keys():
            feature = open_feature(path + str(obj_id) + '.txt')
            feature_dict[obj_id] = feature

        if sub_id not in feature_dict.keys():
            feature = open_feature(path + str(sub_id) + '.txt')
            feature_dict[sub_id] = feature
        # union_id = info['union_boxes_id']
        # if union_id not in feature_dict.keys():
        #     feature = open_feature(path + str(union_id) + '.txt')
        #     feature_dict[union_id] = feature
    #
    # list_file = os.listdir(path)
    # for file in tqdm(list_file):
    #     feature = open_feature(path + file)
    #     feature_dict[file.replace('.txt', '')] = feature
    return feature_dict

def load_union_feature(path, json_path):
    feature_dict = dict()
    json_file = json.load(open(json_path, 'r'))
    for info in tqdm(json_file):
        # obj_id = info['object']
        # sub_id = info['subject']
        # if obj_id not in feature_dict.keys():
        #     feature = open_feature(path + str(obj_id) + '.txt')
        #     feature_dict[obj_id] = feature
        #
        # if sub_id not in feature_dict.keys():
        #     feature = open_feature(path + str(sub_id) + '.txt')
        #     feature_dict[sub_id] = feature
        union_id = info['union_boxes_id']
        if union_id not in feature_dict.keys():
            feature = open_feature(path + str(union_id) + '.txt')
            feature_dict[union_id] = feature
    #
    # list_file = os.listdir(path)
    # for file in tqdm(list_file):
    #     feature = open_feature(path + file)
    #     feature_dict[file.replace('.txt', '')] = feature
    return feature_dict

def open_feature(path):
    f = open(path)
    lines = f.readlines()
    return lines[0]

def save_data(path, dataset_list):
    b = json.dumps(dataset_list)
    f2 = open(path, 'w')
    f2.write(b)
    f2.close()

if __name__ == '__main__':
    # path = '/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/data/boxes_feature/'
    # json_path = '/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/train_vg.json'
    # features = load_feature(path, json_path)
    #
    # save_data('/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/boxes_feature/train_boxes_feature.json', features)
    # path = '/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/data/boxes_feature/'
    json_path = '/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/val_vg.json'
    # features = load_feature(path, json_path)
    #
    # save_data('/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/boxes_feature/val_boxes_feature.json', features)

    union_path = '/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/data/union_boxes/'

    train_json_path = '/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/train_vg.json'
    train_features = load_union_feature(union_path, train_json_path)

    save_data('/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/union_feature/train_union_feature.json', train_features)

    # val_json_path = '/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/val_vg.json'
    val_features = load_union_feature(union_path, json_path)

    save_data('/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/union_feature/val_union_feature.json', val_features)
