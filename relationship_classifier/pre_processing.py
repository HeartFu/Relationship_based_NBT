import json

from tqdm import tqdm


def processing_vg(path):
    data_file = open(path)
    data_lines = data_file.readlines()
    mode_data_list = []
    for data_line in data_lines:
        data_line_split = data_line.split(' ')
        img_path = data_line_split[0].split('/')
        # folder = img_path[0]
        img_name = int(img_path[1].replace('.jpg', ''))
        mode_data_list.append(img_name)

    return mode_data_list


def save_data(path, dataset_list):
    b = json.dumps(dataset_list)
    f2 = open(path, 'w')
    f2.write(b)
    f2.close()


if __name__ == '__main__':
    train_data_path = 'data/train.txt'
    train_data_list = processing_vg(train_data_path)

    val_data_path = 'data/val.txt'
    val_data_list = processing_vg(val_data_path)

    test_data_path = 'data/test.txt'
    test_data_list = processing_vg(test_data_path)

    json_path = '/home/fanfu/newdisk/pytorch-bottom-up-attention/py-bottom-up-attention/demo/relationship_classifier/data/union_boxes/union_boxes_info.json'
    json_data = json.load(open(json_path, 'r'))

    train_dataset = []
    val_dataset = []
    test_dataset = []
    for dataset in tqdm(json_data):
        img_id = dataset['image_id']
        if img_id in train_data_list:
            train_dataset.append(dataset)
        elif img_id in val_data_list:
            val_dataset.append(dataset)
        elif img_id in test_data_list:
            test_dataset.append(dataset)
        else:
            continue

    print('train dataset length: {}'.format(len(train_dataset)))
    print('val dataset length: {}'.format(len(val_dataset)))
    print('test dataset length: {}'.format(len(test_dataset)))

    save_data('data/train_vg.json', train_dataset)

    save_data('data/val_vg.json', val_dataset)

    save_data('data/test_vg.json', test_dataset)
