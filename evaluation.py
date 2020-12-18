from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import torch.backends.cudnn as cudnn

import opts
from eval_tools import eval_NBT, eval_fusion_models
from misc import utils, AttModel
import yaml


def eval_nbt(opt):

    cudnn.benchmark = True

    from misc.dataloader_coco import DataLoader

    ####################################################################################
    # Data Loader
    ####################################################################################
    dataset_val = DataLoader(opt, split=opt.val_split)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
                                                 shuffle=False, num_workers=1)

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
    if opt.att_model == 'topdown':
        model = AttModel.TopDownModel(opt)
    else:
        model = AttModel.Att2in2Model(opt)

    if opt.start_from is not None:
        if opt.load_best_score == 1:
            model_path = os.path.join(opt.start_from, 'model-best.pth')
            info_path = os.path.join(opt.start_from, 'infos_' + opt.id + '-best.pkl')
        else:
            model_path = os.path.join(opt.start_from, 'model.pth')
            info_path = os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')

        # opt.learning_rate = saved_model_opt.learning_rate
        print('Loading the model weights, path is %s...' % (model_path))
        model.load_state_dict(torch.load(model_path))

    if opt.mGPUs:
        model = nn.DataParallel(model)

    if opt.cuda:
        model.cuda()

    ####################################################################################
    # Evaluate the model
    ####################################################################################
    lang_stats, predictions = eval_NBT(opt, model, dataset_val, processing='eval')

    print('print the evaluation:')
    for k, v in lang_stats.items():
        print('{}:{}'.format(k, v))

    # print('predictions: {}'.format(predictions))
    # print({k: v for k, v in lang_stats.items()})


def build_model(opt, start_from):
    if opt.att_model == 'topdown':
        model = AttModel.TopDownModel(opt)
    else:
        model = AttModel.Att2in2Model(opt)

    if start_from is not None:
        if opt.load_best_score == 1:
            model_path = os.path.join(start_from, 'model-best.pth')
            info_path = os.path.join(start_from, 'infos_' + opt.id + '-best.pkl')
        else:
            model_path = os.path.join(start_from, 'model.pth')
            info_path = os.path.join(start_from, 'infos_' + opt.id + '.pkl')

        # opt.learning_rate = saved_model_opt.learning_rate
        print('Loading the model weights, path is %s...' % (model_path))
        model.load_state_dict(torch.load(model_path))

    if opt.mGPUs:
        model = nn.DataParallel(model)

    if opt.cuda:
        model.cuda()

    return model


def eval_relationNBT(opt):

    cudnn.benchmark = True

    if opt.imp_pro == 0.0 and opt.spa_pro == 0.0 and opt.sem_pro == 0.0:
        # no relation module in this pre-trained model
        eval_nbt(opt)
        return

    from misc.dataloader_coco import DataLoader

    ####################################################################################
    # Data Loader
    ####################################################################################
    dataset_val = DataLoader(opt, split=opt.val_split)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
                                                 shuffle=False, num_workers=1)

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
    if opt.imp_model:
        opt.relation_type = 'implicit'
        imp_model = build_model(opt, opt.imp_start_from)
        imp_model.eval()
    else:
        imp_model = None

    if opt.spa_model:
        opt.relation_type = 'spatial'
        spa_model = build_model(opt, opt.spa_start_from)
        spa_model.eval()
    else:
        spa_model = None

    if opt.sem_model:
        opt.relation_type = 'semantic'
        sem_model = build_model(opt, opt.sem_start_from)
        sem_model.eval()
    else:
        sem_model = None

    ####################################################################################
    # Evaluate the model
    ####################################################################################

    lang_stats, predictions = eval_fusion_models(opt, dataset_val, opt.imp_pro, opt.spa_pro, opt.sem_pro, imp_model, spa_model, sem_model)

    print('print the evaluation:')
    for k, v in lang_stats.items():
        print('{}:{}'.format(k, v))

    # print('predictions: {}'.format(predictions))
    # print({k: v for k, v in lang_stats.items()})

# CUDA_VISIBLE_DEVICES=2 python evaluation.py --path_opt cfgs/normal_coco_res101.yml --batch_size 100 --cuda True --num_workers 1 --beam_size 3 --start_from save/bs100_semantic --relation_type semantic
# CUDA_VISIBLE_DEVICES=2 python evaluation.py --path_opt cfgs/normal_coco_res101.yml --batch_size 1 --cuda True --num_workers 1 --beam_size 1 --sem_model --spa_model --imp_model --sem_pro 0.3 --spa_pro 0.3 --imp_pro 0.4
if __name__ == '__main__':
    opt = opts.parse_opt()
    if opt.path_opt is not None:
        with open(opt.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
        utils.update_values(options_yaml, vars(opt))
    print(opt)

    if opt.imp_model or opt.sem_model or opt.spa_model:
        eval_relationNBT(opt)
    else:
        eval_nbt(opt)
