from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import torch.backends.cudnn as cudnn

import opts
from eval_tools import demo_fusion_models
from evaluation import build_model
from misc import utils
import yaml

def demo_relationNBT(opt):
    cudnn.benchmark = True

    from misc.dataloader_coco import DataLoader

    ####################################################################################
    # Data Loader
    ####################################################################################
    dataset_val = DataLoader(opt, split=opt.val_split)
    # dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
    #                                              shuffle=False, num_workers=1)

    input_imgs = torch.FloatTensor(1)
    input_seqs = torch.LongTensor(1)
    input_ppls = torch.FloatTensor(1)
    gt_bboxs = torch.FloatTensor(1)
    mask_bboxs = torch.ByteTensor(1)
    gt_seqs = torch.LongTensor(1)
    input_num = torch.LongTensor(1)

    if opt.cuda:
        input_imgs = input_imgs.cuda()
        input_seqs = input_seqs.cuda()
        gt_seqs = gt_seqs.cuda()
        input_num = input_num.cuda()
        input_ppls = input_ppls.cuda()
        gt_bboxs = gt_bboxs.cuda()
        mask_bboxs = mask_bboxs.cuda()

    input_imgs = Variable(input_imgs)
    input_seqs = Variable(input_seqs)
    gt_seqs = Variable(gt_seqs)
    input_num = Variable(input_num)
    input_ppls = Variable(input_ppls)
    gt_bboxs = Variable(gt_bboxs)
    mask_bboxs = Variable(mask_bboxs)

    ####################################################################################
    # Build the Model
    ####################################################################################
    opt.vocab_size = dataset_val.vocab_size
    opt.detect_size = dataset_val.detect_size
    opt.seq_length = opt.seq_length
    opt.fg_size = dataset_val.fg_size
    opt.fg_mask = torch.from_numpy(dataset_val.fg_mask).byte()
    opt.glove_fg = torch.from_numpy(dataset_val.glove_fg).float()
    opt.glove_clss = torch.from_numpy(dataset_val.glove_clss).float()
    opt.glove_w = torch.from_numpy(dataset_val.glove_w).float()
    opt.st2towidx = torch.from_numpy(dataset_val.st2towidx).long()

    opt.itow = dataset_val.itow
    opt.itod = dataset_val.itod
    opt.ltow = dataset_val.ltow
    opt.itoc = dataset_val.itoc

    # choose the attention model
    save_name = ''
    if opt.imp_model:
        opt.relation_type = 'implicit'
        imp_model = build_model(opt, opt.imp_start_from)
        imp_model.eval()
        save_name += '_imp'
    else:
        imp_model = None

    if opt.spa_model:
        opt.relation_type = 'spatial'
        spa_model = build_model(opt, opt.spa_start_from)
        spa_model.eval()
        save_name += '_spa'
    else:
        spa_model = None

    if opt.sem_model:
        opt.relation_type = 'semantic'
        sem_model = build_model(opt, opt.sem_start_from)
        sem_model.eval()
        save_name += '_sem'
    else:
        sem_model = None

    ####################################################################################
    # Evaluate the model
    ####################################################################################
    predictions = demo_fusion_models(opt, dataset_val, opt.imp_pro, opt.spa_pro, opt.sem_pro, imp_model, spa_model, sem_model, save_name)
    print('saving...')
    json.dump(predictions, open('visu/visu_relation'+ save_name + '.json', 'w'))


# CUDA_VISIBLE_DEVICES=1 python demo_relation_nbt.py --path_opt cfgs/normal_coco_res101.yml --batch_size 100 --cuda True --num_workers 1 --beam_size 3 --sem_model --spa_model --imp_model --sem_pro 0.3 --spa_pro 0.3 --imp_pro 0.4
if __name__ == '__main__':
    opt = opts.parse_opt()
    if opt.path_opt is not None:
        with open(opt.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
        utils.update_values(options_yaml, vars(opt))
    print(opt)

    demo_relationNBT(opt)
