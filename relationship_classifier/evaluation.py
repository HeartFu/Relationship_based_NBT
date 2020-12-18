import os
import time

import torch
import wandb
from torch.optim import lr_scheduler
from torch.utils import data
import torch.optim as optim
from tqdm import tqdm
import sys
import torch.nn as nn
import opts, utils
from dataset import BoxesDataset
from model.Classifier import Classifier


def evaluation(cfgs):
    # dataset

    # train_dataset = BoxesDataset(os.path.join(cfgs.data_path, 'train_vg.json'), cfgs.boxes_path, cfgs.union_path)
    # train_loader = data.DataLoader(dataset=train_dataset, batch_size=cfgs.batch_size, num_workers=1, shuffle=False)
    #
    # val_dataset = BoxesDataset(os.path.join(cfgs.data_path, 'val_vg.json'), cfgs.boxes_path, cfgs.union_path)
    # val_loader = data.DataLoader(dataset=val_dataset, batch_size=cfgs.batch_size, shuffle=False)
    test_dataset = BoxesDataset(os.path.join(cfgs.data_path, 'test_union_info.json'),
                                 os.path.join(cfgs.data_path, 'test_union_features.json'),
                                 os.path.join(cfgs.data_path, 'test_boxes_features.json'))
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, num_workers=1, shuffle=False)
    #
    # val_dataset = BoxesDataset(os.path.join(cfgs.data_path, 'val_union_info.json'),
    #                              os.path.join(cfgs.data_path, 'val_union_features.json'),
    #                              os.path.join(cfgs.data_path, 'val_boxes_features.json'))
    # val_loader = data.DataLoader(dataset=val_dataset, batch_size=cfgs.batch_size, num_workers=1, shuffle=False)

    # Model
    model = Classifier()

    checkpoint = torch.load(cfgs.checkpoint + 'checkpoint_final.pkl')
    model.load_state_dict(checkpoint['model_state_dict'])

    # if cfgs.resume:
    #     checkpoint = torch.load(cfgs.checkpoint + 'checkpoint_final.pkl')
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     epoch = checkpoint['epoch']
    #     learning_rate = checkpoint['learning_rate']
    #     train_loss_epoch = checkpoint['train_loss_epoch']
    #     train_acc_epoch = checkpoint['train_acc_epoch']
    #     test_acc_epoch = checkpoint['test_acc_epoch']
    # else:
    #     epoch = 0
    #     learning_rate = cfgs.learning_rate
    #     train_loss_epoch = []
    #     train_acc_epoch = []
    #     test_acc_epoch = []

    if cfgs.mGPUs:
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    # if (epoch + 1) % 5 == 0:
    with torch.no_grad():
        test_total = 0
        test_correct = 0
        process_bar_test = tqdm(test_loader, desc='|Test Evaluation', leave=False)
        for i, batch in enumerate(process_bar_test):
            obj_feat, sub_feat, union_feat, labels = batch
            obj_feat, sub_feat, union_feat, labels = obj_feat.cuda(), sub_feat.cuda(), union_feat.cuda(), labels.cuda()
            outputs = model(obj_feat, sub_feat, union_feat)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            correct = torch.sum(predicted == labels).item()
            test_correct += correct
            process_bar_test.set_postfix({'test_accuracy': '{:.3f}'.format(float(correct / labels.size(0)))},
                                         refresh=True)

        test_acc_aveg = float(test_correct) / test_total


        print('Accuracy of the model on testset: {}'.format(test_acc_aveg))
        # test_acc_epoch.append(test_acc_aveg)


if __name__ == '__main__':
    # wandb.init(project="relationship_classifier")
    opt = opts.parse_opt()
    # sys.stdout = utils.Logger(opt.output_dir + 'classifier/info_weighNorm.log', sys.stdout)
    # sys.stderr = utils.Logger(opt.output_dir + 'classifier/error_weighNorm.log', sys.stderr)
    evaluation(opt)
