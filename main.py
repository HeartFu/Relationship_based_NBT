from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import time
import os
import pickle
import torch.backends.cudnn as cudnn
import yaml
from tqdm import tqdm

import opts
from eval_tools import eval_NBT
from misc import utils, eval_utils, AttModel
import yaml

import wandb

# from misc.rewards import get_self_critical_reward
import torchvision.transforms as transforms
import pdb


from relation_utils import prepare_graph_variables


def train(epoch, opt):
    model.train()

    #########################################################################################
    # Training begins here
    #########################################################################################
    # data_iter = iter(dataloader)
    lm_loss_temp = 0
    bn_loss_temp = 0
    fg_loss_temp = 0
    total_loss_temp = 0
    cider_temp = 0
    rl_loss_temp = 0
    start = time.time()
    # for step in range(len(dataloader)-1):
    # data = data_iter.next()
    count = 0
    progress_bar = tqdm(dataloader, desc='|Train Epoch {}'.format(epoch), leave=False)
    for step, data in enumerate(progress_bar):
        count += 1
        img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id, spa_adj_matrix, sem_adj_matrix = data
        # if img_id.item() == 262284:
        #     import pdb
        #     pdb.set_trace()
        # print("images ids: {}".format(img_id))
        proposals = proposals[:, :max(int(max(num[:, 1])), 1), :]
        bboxs = bboxs[:, :int(max(num[:, 2])), :]
        box_mask = box_mask[:, :, :max(int(max(num[:, 2])), 1), :]
        if len(spa_adj_matrix[0]) != 0:
            spa_adj_matrix = spa_adj_matrix[:, :max(int(max(num[:, 1])), 1), :max(int(max(num[:, 1])), 1)]
        if len(sem_adj_matrix[0]) != 0:
            sem_adj_matrix = sem_adj_matrix[:, :max(int(max(num[:, 1])), 1), :max(int(max(num[:, 1])), 1)]

        # with torch.no_grad():
        input_imgs.resize_(img.size()).copy_(img)
        input_seqs.resize_(iseq.size()).copy_(iseq)
        gt_seqs.resize_(gts_seq.size()).copy_(gts_seq)
        input_num.resize_(num.size()).copy_(num)
        input_ppls.resize_(proposals.size()).copy_(proposals)
        gt_bboxs.resize_(bboxs.size()).copy_(bboxs)
        mask_bboxs.resize_(box_mask.size()).copy_(box_mask)

        # mask_bboxs.resize_(box_mask.size()).copy_(box_mask)

        # mask_bboxs = mask_bboxs.bool()
        # input_imgs.data.resize_(img.size()).copy_(img)
        # input_seqs.data.resize_(iseq.size()).copy_(iseq)
        # gt_seqs.data.resize_(gts_seq.size()).copy_(gts_seq)
        # input_num.data.resize_(num.size()).copy_(num)
        # input_ppls.data.resize_(proposals.size()).copy_(proposals)
        # gt_bboxs.data.resize_(bboxs.size()).copy_(bboxs)
        # mask_bboxs.data.resize_(box_mask.size()).copy_(box_mask)
        loss = 0

        # relationship modify
        eval_opt = {
            'imp_model': False,
            'spa_model': False,
            'sem_model': False,
            'graph_att': opt.graph_attention,
        }
        pos_emb_var, spa_adj_matrix, sem_adj_matrix = prepare_graph_variables(opt.relation_type, proposals[:, :, :4], sem_adj_matrix, spa_adj_matrix,
                                                              opt.nongt_dim, opt.imp_pos_emb_dim, opt.spa_label_num,
                                                              opt.sem_label_num, eval_opt)
        # print(spa_adj_matrix.size())
        # print(input_ppls.size())

        if opt.self_critical:
            rl_loss, bn_loss, fg_loss, cider_score = model(input_imgs, input_seqs, gt_seqs, input_num, input_ppls,
                                                           gt_bboxs, mask_bboxs, 'RL')
            cider_temp += cider_score.sum().data[0] / cider_score.numel()
            loss += (rl_loss.sum() + bn_loss.sum() + fg_loss.sum()) / rl_loss.numel()
            rl_loss_temp += loss.item()

        else:
            lm_loss, bn_loss, fg_loss = model(input_imgs, input_seqs, gt_seqs, input_num, input_ppls, gt_bboxs,
                                              mask_bboxs, 'MLE', pos_emb_var, spa_adj_matrix, sem_adj_matrix)
            loss += (lm_loss.sum() + bn_loss.sum() + fg_loss.sum()) / lm_loss.numel()

            # FF: modify data[0] to tensor.item()
            lm_loss_temp += lm_loss.sum().item() / lm_loss.numel()
            bn_loss_temp += bn_loss.sum().item() / lm_loss.numel()
            fg_loss_temp += fg_loss.sum().item() / lm_loss.numel()
            total_loss_temp += loss.item()
            # lm_loss_temp += lm_loss.sum().data[0] / lm_loss.numel()
            # bn_loss_temp += bn_loss.sum().data[0] / lm_loss.numel()
            # fg_loss_temp += fg_loss.sum().data[0] / lm_loss.numel()

        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip)
        # utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()

        # if opt.finetune_cnn:
        #     utils.clip_gradient(cnn_optimizer, opt.grad_clip)
        #     cnn_optimizer.step()

        progress_bar.set_postfix({'lm_loss': '{:.3f}'.format(lm_loss.sum().item() / lm_loss.numel()),
                                  'bn_loss': '{:.3f}'.format(bn_loss.sum().item() / lm_loss.numel()),
                                  'fg_loss': '{:.3f}'.format(fg_loss.sum().item() / lm_loss.numel()),
                                  'total_loss': '{:.3f}'.format(loss.item()),
                                  'rl_loss': '{:.3f}'.format(rl_loss_temp / count),
                                  'cider_score': '{:.3f}'.format(cider_temp / count),
                                  'lr': '{:.5f}'.format(opt.learning_rate),
                                  },
                                 refresh=True)

        # if step % opt.disp_interval == 0 and step != 0:
        #     end = time.time()
        #     lm_loss_temp /= opt.disp_interval
        #     bn_loss_temp /= opt.disp_interval
        #     fg_loss_temp /= opt.disp_interval
        #     rl_loss_temp /= opt.disp_interval
        #
        #     cider_temp /= opt.disp_interval
        #     print(
        #         "step {}/{} (epoch {}), lm_loss = {:.3f}, bn_loss = {:.3f}, fg_loss = {:.3f}, rl_loss = {:.3f}, cider_score = {:.3f}, lr = {:.5f}, time/batch = {:.3f}" \
        #         .format(step, len(dataloader), epoch, lm_loss_temp, bn_loss_temp, fg_loss_temp, rl_loss_temp,
        #                 cider_temp, opt.learning_rate, end - start))
        #     start = time.time()
        #
        #     lm_loss_temp = 0
        #     bn_loss_temp = 0
        #     fg_loss_temp = 0
        #     cider_temp = 0
        #     rl_loss_temp = 0
        #     count = 0

        # Write the training loss summary
        if iteration % opt.losses_log_every == 0:

            # use wand to observe the loss and lr
            if wandb is not None:
                wandb.log({
                    'lm_loss': lm_loss.sum().item() / lm_loss.numel(),
                    'bn_loss': bn_loss.sum().item() / lm_loss.numel(),
                    'fg_loss': fg_loss.sum().item() / lm_loss.numel(),
                    'total_loss': loss,
                })

            loss_history[iteration] = loss.item()
            # loss_history[iteration] = loss.data[0]
            lr_history[iteration] = opt.learning_rate
            # ss_prob_history[iteration] = model.ss_prob

        # delete empty cache for release some GPU memory
        del pos_emb_var, spa_adj_matrix, sem_adj_matrix
        torch.cuda.empty_cache()

    end = time.time()
    print(
        "epoch: {}, lm_loss = {:.3f}, bn_loss = {:.3f}, fg_loss = {:.3f}, rl_loss = {:.3f}, cider_score = {:.3f}, lr = {:.5f}, time/batch = {:.3f}" \
        .format(epoch, lm_loss_temp / count, bn_loss_temp / count, fg_loss_temp / count, rl_loss_temp / count,
                cider_temp / count, opt.learning_rate, end - start))
    # return lm_loss_temp/count, bn_loss_temp/count, fg_loss_temp/count, total_loss_temp/count
    if wandb is not None:
        wandb.log({
            'lm_loss_epoch': lm_loss_temp / count / lm_loss.numel(),
            'bn_loss_epoch': bn_loss_temp / count,
            'fg_loss_epoch': fg_loss_temp / count,
            'total_loss_epoch': total_loss_temp / count,
        })




####################################################################################
# Main
####################################################################################
# initialize the data holder.
#python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 20 --cuda True --num_workers 4 --max_epoch 50 --proposal_h5 detector.json --mGPUs True
# python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 100 --cuda True --num_workers 10 --max_epoch 50 --proposal_h5 detector.json --mGPUs True --graph_attention True
if __name__ == '__main__':
    opt = opts.parse_opt()
    if opt.path_opt is not None:
        with open(opt.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
        utils.update_values(options_yaml, vars(opt))
    print(opt)
    debug = True
    if not debug:
        wandb.init(project="neural_baby_talk")
    else:
        wandb = None

    cudnn.benchmark = True
    # print(opt.proposal_h5)

    from misc.dataloader_coco import DataLoader

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    ####################################################################################
    # Data Loader
    ####################################################################################
    dataset = DataLoader(opt, split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=False, num_workers=opt.num_workers)
    # import pdb
    # pdb.set_trace()
    dataset_val = DataLoader(opt, split=opt.val_split)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size,
                                                 shuffle=False, num_workers=opt.num_workers)

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
    opt.vocab_size = dataset.vocab_size
    opt.detect_size = dataset.detect_size
    opt.seq_length = opt.seq_length
    opt.fg_size = dataset.fg_size
    opt.fg_mask = torch.from_numpy(dataset.fg_mask).byte()
    opt.glove_fg = torch.from_numpy(dataset.glove_fg).float()
    opt.glove_clss = torch.from_numpy(dataset.glove_clss).float()
    opt.glove_w = torch.from_numpy(dataset.glove_w).float()
    opt.st2towidx = torch.from_numpy(dataset.st2towidx).long()

    opt.itow = dataset.itow
    opt.itod = dataset.itod
    opt.ltow = dataset.ltow
    opt.itoc = dataset.itoc

    if not opt.finetune_cnn: opt.fixed_block = 4  # if not finetune, fix all cnn block

    if opt.att_model == 'topdown':
        model = AttModel.TopDownModel(opt)
    elif opt.att_model == 'att2in2':
        model = AttModel.Att2in2Model(opt)

    # tf_summary_writer = tf and tf.compat.v1.summary.FileWriter(opt.checkpoint_path)
    infos = {}
    histories = {}
    if opt.start_from is not None:
        if opt.load_best_score == 1:
            model_path = os.path.join(opt.start_from, 'model-best.pth')
            info_path = os.path.join(opt.start_from, 'infos_' + opt.id + '-best.pkl')
        else:
            model_path = os.path.join(opt.start_from, 'model.pth')
            info_path = os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')

            # open old infos and check if models are compatible
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            saved_model_opt = infos['opt']

        opt.learning_rate = saved_model_opt.learning_rate
        print('Loading the model %s...' % (model_path))
        model.load_state_dict(torch.load(model_path))

        if os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
            with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
                histories = pickle.load(f)

    if opt.decode_noc:
        model._reinit_word_weight(opt, dataset.ctoi, dataset.wtoi)

    best_val_score = infos.get('best_val_score', None)
    iteration = infos.get('iter', 0)
    start_epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    if opt.mGPUs:
        model = nn.DataParallel(model)

    if opt.cuda:
        model.cuda()

    params = []
    # cnn_params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'cnn' in key:
                params += [{'params': [value], 'lr': opt.cnn_learning_rate,
                            'weight_decay': opt.cnn_weight_decay, 'betas': (opt.cnn_optim_alpha, opt.cnn_optim_beta)}]
            else:
                params += [{'params': [value], 'lr': opt.learning_rate,
                            'weight_decay': opt.weight_decay, 'betas': (opt.optim_alpha, opt.optim_beta)}]

    print("Use %s as optmization method" % (opt.optim))
    if opt.optim == 'sgd':
        optimizer = optim.SGD(params, momentum=0.9)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(params)
    elif opt.optim == 'adamax':
        optimizer = optim.Adamax(params)

    # if opt.cnn_optim == 'sgd':
    #     cnn_optimizer = optim.SGD(cnn_params, momentum=0.9)
    # else:
    #     cnn_optimizer = optim.Adam(cnn_params)
    # load optimizer
    # learning_rate_list = np.linspace(opt.learning_rate, 0.0005, opt.max_epochs)

    # monitor the model with wandb
    if wandb is not None:
        wandb.watch(model)

    print(model)

    for epoch in range(start_epoch, opt.max_epochs):
        if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
            if (epoch - opt.learning_rate_decay_start) % opt.learning_rate_decay_every == 0:
                # decay the learning rate.
                utils.set_lr(optimizer, opt.learning_rate_decay_rate)
                opt.learning_rate = opt.learning_rate * opt.learning_rate_decay_rate

        if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
            model.ss_prob = opt.ss_prob

        if not opt.inference_only:
            train(epoch, opt)

        if (epoch + 1) % opt.val_every_epoch == 0:
            lang_stats, predictions = eval_NBT(opt, model, dataset_val)
            val_result_history[iteration] = {'lang_stats': lang_stats, 'predictions': predictions}
            if wandb is not None:
                wandb.log({k: v for k, v in lang_stats.items()})
            # Save model if is improving on validation result
            current_score = lang_stats['CIDEr']

            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
            if opt.mGPUs:
                torch.save(model.module.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
            # optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            # torch.save(optimizer.state_dict(), optimizer_path)

            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = dataset.itow

            histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['lr_history'] = lr_history
            histories['ss_prob_history'] = ss_prob_history
            with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '.pkl'), 'wb') as f:
                pickle.dump(infos, f)
            with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + '.pkl'), 'wb') as f:
                pickle.dump(histories, f)

            if best_flag:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                if opt.mGPUs:
                    torch.save(model.module.state_dict(), checkpoint_path)
                else:
                    torch.save(model.state_dict(), checkpoint_path)

                print("model saved to {} with best cider score {:.3f}".format(checkpoint_path, best_val_score))
                with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '-best.pkl'), 'wb') as f:
                    pickle.dump(infos, f)
