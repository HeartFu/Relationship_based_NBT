import json
import os
import time

import torch
from torch.autograd import Variable
from tqdm import tqdm
from misc import utils
from relation_utils import prepare_graph_variables
import torch.nn.functional as F
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


def eval_NBT(opt, model, dataset_val, processing='train'):
    model.eval()
    #########################################################################################
    # eval begins here
    #########################################################################################
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

    data_iter_val = iter(dataloader_val)
    loss_temp = 0
    start = time.time()

    num_show = 0
    predictions = []
    progress_bar = tqdm(dataloader_val, desc='|Validation process', leave=False)
    # for step in range(len(dataloader_val)):
    for step, data in enumerate(progress_bar):
        # data = data_iter_val.next()
        img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id, spa_adj_matrix, sem_adj_matrix = data

        proposals = proposals[:, :max(int(max(num[:, 1])), 1), :]

        # FF: Fix the bug with .data not run in the Pytorch
        input_imgs.resize_(img.size()).copy_(img)
        input_seqs.resize_(iseq.size()).copy_(iseq)
        gt_seqs.resize_(gts_seq.size()).copy_(gts_seq)
        input_num.resize_(num.size()).copy_(num)
        input_ppls.resize_(proposals.size()).copy_(proposals)
        gt_bboxs.resize_(bboxs.size()).copy_(bboxs)
        # FF: modify 0/1 to true/false
        mask_bboxs.resize_(box_mask.size()).copy_(box_mask.bool())
        # mask_bboxs.data.resize_(box_mask.size()).copy_(box_mask)
        input_imgs.resize_(img.size()).copy_(img)

        if len(spa_adj_matrix[0]) != 0:
            spa_adj_matrix = spa_adj_matrix[:, :max(int(max(num[:, 1])), 1), :max(int(max(num[:, 1])), 1)]
        if len(sem_adj_matrix[0]) != 0:
            sem_adj_matrix = sem_adj_matrix[:, :max(int(max(num[:, 1])), 1), :max(int(max(num[:, 1])), 1)]

        # relationship modify
        # eval_opt_rel = {
        #     "graph_att": opt.graph_attention
        # }
        eval_opt_rel = {
            'imp_model': False,
            'spa_model': False,
            'sem_model': False,
            "graph_att": opt.graph_attention
        }

        pos_emb_var, spa_adj_matrix, sem_adj_matrix = prepare_graph_variables(opt.relation_type, proposals[:, :, :4],
                                                                              sem_adj_matrix, spa_adj_matrix,
                                                                              opt.nongt_dim, opt.imp_pos_emb_dim,
                                                                              opt.spa_label_num,
                                                                              opt.sem_label_num, eval_opt_rel)

        eval_opt = {'sample_max': 1, 'beam_size': opt.beam_size, 'inference_mode': True, 'tag_size': opt.cbs_tag_size}
        if processing == 'train':
            seq, bn_seq, fg_seq = model(input_imgs, input_seqs, gt_seqs, \
                                        input_num, input_ppls, gt_bboxs, mask_bboxs, 'sample', pos_emb_var,
                                        spa_adj_matrix,
                                        sem_adj_matrix, eval_opt)
        else:
            seq, bn_seq, fg_seq, seqLogprobs, bnLogprobs, fgLogprobs = model._sample(input_imgs, input_ppls, input_num,
                                                                                     pos_emb_var, spa_adj_matrix,
                                                                                     sem_adj_matrix, eval_opt)
            # import pdb
            # pdb.set_trace()
        sents = utils.decode_sequence(dataset_val.itow, dataset_val.itod, dataset_val.ltow, dataset_val.itoc,
                                      dataset_val.wtod, \
                                      seq.data, bn_seq.data, fg_seq.data, opt.vocab_size, opt)
        for k, sent in enumerate(sents):
            entry = {'image_id': img_id[k].item(), 'caption': sent}
            predictions.append(entry)
            if num_show < 20:
                print('image %s: %s' % (entry['image_id'], entry['caption']))
                num_show += 1

    print('Total image to be evaluated %d' % (len(predictions)))
    lang_stats = None
    if opt.language_eval == 1:
        if opt.decode_noc:
            lang_stats = utils.noc_eval(predictions, str(1), opt.val_split, opt)
        else:
            lang_stats = utils.language_eval(opt.dataset, predictions, str(1), opt.val_split, opt)

    print('Saving the predictions')

    # Write validation result into summary
    # if tf is not None:
    #     for k, v in lang_stats.items():
    #         add_summary_value(tf_summary_writer, k, v, iteration)
    #     tf_summary_writer.flush()

    # TODO: change the train process
    # val_result_history[iteration] = {'lang_stats': lang_stats, 'predictions': predictions}
    # if wandb is not None:
    #     wandb.log({k: v for k, v in lang_stats.items()})
    return lang_stats, predictions


def get_rnn_output(model, beam_size, fc_feats, conv_feats, pool_feats, p_conv_feats, p_pool_feats, ppls, rois_num,
                   pnt_mask, k):
    state = model.init_hidden(beam_size)
    beam_fc_feats = fc_feats[k:k + 1].expand(beam_size, fc_feats.size(1))
    beam_conv_feats = conv_feats[k:k + 1].expand(beam_size, conv_feats.size(1), model.rnn_size).contiguous()
    beam_pool_feats = pool_feats[k:k + 1].expand(beam_size, rois_num, model.rnn_size).contiguous()
    beam_p_conv_feats = p_conv_feats[k:k + 1].expand(beam_size, conv_feats.size(1),
                                                     model.att_hid_size).contiguous()
    beam_p_pool_feats = p_pool_feats[k:k + 1].expand(beam_size, rois_num, model.att_hid_size).contiguous()

    beam_ppls = ppls[k:k + 1].expand(beam_size, rois_num, 6).contiguous()
    beam_pnt_mask = pnt_mask[k:k + 1].expand(beam_size, rois_num + 1).contiguous()

    it = fc_feats.data.new(beam_size).long().zero_()
    xt = model.embed(Variable(it))

    rnn_output, det_prob, state = model.core(xt, beam_fc_feats, beam_conv_feats, beam_p_conv_feats, \
                                             beam_pool_feats, beam_p_pool_feats, beam_pnt_mask, beam_pnt_mask,
                                             state)

    return rnn_output, det_prob, state, beam_ppls, beam_pnt_mask, beam_pool_feats, beam_p_pool_feats, \
           beam_fc_feats, beam_conv_feats, beam_p_conv_feats


def demo_fusion_models(opt, dataset_val, imp_pro, spa_pro, sem_pro, imp_model=None, spa_model=None, sem_model=None, save_name=''):
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

    data_iter_val = iter(dataloader_val)
    loss_temp = 0
    start = time.time()

    num_show = 0
    predictions = []
    progress_bar = tqdm(dataloader_val, desc='|Validation process', leave=False)
    # for step in range(len(dataloader_val)):
    for step, data in enumerate(progress_bar):
        if step * opt.batch_size > 1000:
            break
        # data = data_iter_val.next()
        img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id, spa_adj_matrix, sem_adj_matrix = data

        proposals = proposals[:, :max(int(max(num[:, 1])), 1), :]

        # FF: Fix the bug with .data not run in the Pytorch
        input_imgs.resize_(img.size()).copy_(img)
        input_seqs.resize_(iseq.size()).copy_(iseq)
        gt_seqs.resize_(gts_seq.size()).copy_(gts_seq)
        input_num.resize_(num.size()).copy_(num)
        input_ppls.resize_(proposals.size()).copy_(proposals)
        gt_bboxs.resize_(bboxs.size()).copy_(bboxs)
        # FF: modify 0/1 to true/false
        mask_bboxs.resize_(box_mask.size()).copy_(box_mask.bool())
        # mask_bboxs.data.resize_(box_mask.size()).copy_(box_mask)
        input_imgs.resize_(img.size()).copy_(img)

        if len(spa_adj_matrix[0]) != 0:
            spa_adj_matrix = spa_adj_matrix[:, :max(int(max(num[:, 1])), 1), :max(int(max(num[:, 1])), 1)]
        if len(sem_adj_matrix[0]) != 0:
            sem_adj_matrix = sem_adj_matrix[:, :max(int(max(num[:, 1])), 1), :max(int(max(num[:, 1])), 1)]

        # relationship modify
        eval_opt_rel = {
            'imp_model': opt.imp_model,
            'spa_model': opt.spa_model,
            'sem_model': opt.sem_model,
            "graph_att": opt.graph_attention
        }
        pos_emb_var, spa_adj_matrix, sem_adj_matrix = prepare_graph_variables(opt.relation_type, proposals[:, :, :4],
                                                                              sem_adj_matrix, spa_adj_matrix,
                                                                              opt.nongt_dim, opt.imp_pos_emb_dim,
                                                                              opt.spa_label_num, opt.sem_label_num,
                                                                              eval_opt_rel)

        eval_opt = {'sample_max': 1, 'beam_size': opt.beam_size, 'inference_mode': True, 'tag_size': opt.cbs_tag_size}
        seq, bn_seq, fg_seq, seqLogprobs, bnLogprobs, fgLogprobs, _ = fusion_beam_sample(opt, imp_pro, spa_pro, sem_pro,
                                                                                      input_ppls, input_imgs, input_num,
                                                                                      pos_emb_var, spa_adj_matrix,
                                                                                      sem_adj_matrix, eval_opt,
                                                                                      imp_model,
                                                                                      spa_model, sem_model)
        sents, det_idx, det_word = utils.decode_sequence_det(dataset_val.itow, dataset_val.itod, dataset_val.ltow,
                                                             dataset_val.itoc,
                                                             dataset_val.wtod, seq.data, bn_seq.data, fg_seq.data,
                                                             opt.vocab_size, opt)

        for i in range(opt.batch_size):
            print(i)

            if os.path.isfile(os.path.join(opt.image_path, 'val2014/COCO_val2014_%012d.jpg' % img_id[i])):
                im2show = Image.open(
                    os.path.join(opt.image_path, 'val2014/COCO_val2014_%012d.jpg' % img_id[i])).convert('RGB')
            else:
                im2show = Image.open(
                    os.path.join(opt.image_path, 'train2014/COCO_train2014_%012d.jpg' % img_id[i])).convert('RGB')

            w, h = im2show.size

            rest_idx = []
            # import pdb
            # pdb.set_trace()
            proposals_one = proposals[i].numpy()
            ppl_mask = np.all(np.equal(proposals_one, 0), axis=1)
            proposals_one = proposals_one[~ppl_mask]
            # if i == 2:

            # det_idx = det_idx[:proposals_one.shape[0]]
            new_det_idx = []
            for j in range(len(det_idx)):
                if det_idx[j] < proposals_one.shape[0] and det_idx[j] not in new_det_idx:
                    new_det_idx.append(det_idx[j])
            det_idx = new_det_idx
            for j in range(proposals_one.shape[0]):
                if j not in det_idx:
                    rest_idx.append(j)

            if len(det_idx) > 0:
                # for visulization

                proposals_one[:, 0] = proposals_one[:, 0] * w / float(opt.image_crop_size)
                proposals_one[:, 2] = proposals_one[:, 2] * w / float(opt.image_crop_size)
                proposals_one[:, 1] = proposals_one[:, 1] * h / float(opt.image_crop_size)
                proposals_one[:, 3] = proposals_one[:, 3] * h / float(opt.image_crop_size)

                cls_dets = proposals_one[det_idx]
                rest_dets = proposals_one[rest_idx]

            fig = plt.figure(frameon=False)
            # fig.set_size_inches(5,5*h/w)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            a = fig.gca()
            a.set_frame_on(False)
            a.set_xticks([])
            a.set_yticks([])
            plt.axis('off')
            plt.xlim(0, w)
            plt.ylim(h, 0)
            # fig, ax = plt.subplots(1)

            # show other box in grey.

            plt.imshow(im2show)

            if len(rest_idx) > 0:
                for j in range(len(rest_dets)):
                    ax = utils.vis_detections(ax, dataset_val.itoc[int(rest_dets[j, 4])], rest_dets[j, :5], j, 1)
            # import pdb
            # pdb.set_trace()
            if len(det_idx) > 0:
                for j in range(len(cls_dets)):
                    ax = utils.vis_detections(ax, dataset_val.itoc[int(cls_dets[j, 4])], cls_dets[j, :5], j, 0)

            # plt.axis('off')
            # plt.axis('tight')
            # plt.tight_layout()
            # import pdb
            # pdb.set_trace()
            # fig.savefig('visu/visu_relation'+ save_name + '/%d.jpg' % (img_id[i].item()),
            #             bbox_inches='tight', pad_inches=0, dpi=150)
            # fig.savefig('visu_relation/%d.jpg' % (img_id[i].item()),
            #             bbox_inches='tight', pad_inches=0, dpi=150)
            print(str(img_id[i].item()) + ': ' + sents[i])

            entry = {'image_id': img_id[i].item(), 'caption': sents[i]}
            predictions.append(entry)

    return predictions


def eval_fusion_models(opt, dataset_val, imp_pro, spa_pro, sem_pro, imp_model=None, spa_model=None, sem_model=None):
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

    data_iter_val = iter(dataloader_val)
    loss_temp = 0
    start = time.time()

    num_show = 0
    predictions = []
    progress_bar = tqdm(dataloader_val, desc='|Validation process', leave=False)
    # for step in range(len(dataloader_val)):
    for step, data in enumerate(progress_bar):
        # data = data_iter_val.next()
        img, iseq, gts_seq, num, proposals, bboxs, box_mask, img_id, spa_adj_matrix, sem_adj_matrix = data
        # print(img_id)
        proposals = proposals[:, :max(int(max(num[:, 1])), 1), :]
        # print(proposals)
        # FF: Fix the bug with .data not run in the Pytorch
        input_imgs.resize_(img.size()).copy_(img)
        input_seqs.resize_(iseq.size()).copy_(iseq)
        gt_seqs.resize_(gts_seq.size()).copy_(gts_seq)
        input_num.resize_(num.size()).copy_(num)
        input_ppls.resize_(proposals.size()).copy_(proposals)
        gt_bboxs.resize_(bboxs.size()).copy_(bboxs)
        # FF: modify 0/1 to true/false
        mask_bboxs.resize_(box_mask.size()).copy_(box_mask.bool())
        # mask_bboxs.data.resize_(box_mask.size()).copy_(box_mask)
        input_imgs.resize_(img.size()).copy_(img)

        if len(spa_adj_matrix[0]) != 0:
            spa_adj_matrix = spa_adj_matrix[:, :max(int(max(num[:, 1])), 1), :max(int(max(num[:, 1])), 1)]
        if len(sem_adj_matrix[0]) != 0:
            sem_adj_matrix = sem_adj_matrix[:, :max(int(max(num[:, 1])), 1), :max(int(max(num[:, 1])), 1)]

        # relationship modify
        eval_opt_rel = {
            'imp_model': opt.imp_model,
            'spa_model': opt.spa_model,
            'sem_model': opt.sem_model,
            "graph_att": opt.graph_attention
        }
        pos_emb_var, spa_adj_matrix, sem_adj_matrix = prepare_graph_variables(opt.relation_type, proposals[:, :, :4],
                                                                              sem_adj_matrix, spa_adj_matrix,
                                                                              opt.nongt_dim, opt.imp_pos_emb_dim,
                                                                              opt.spa_label_num, opt.sem_label_num,
                                                                              eval_opt_rel)

        eval_opt = {'sample_max': 1, 'beam_size': opt.beam_size, 'inference_mode': True, 'tag_size': opt.cbs_tag_size}
        seq, bn_seq, fg_seq, seqLogprobs, bnLogprobs, fgLogprobs, attention_weights = fusion_beam_sample(opt, imp_pro,
                                                                                                         spa_pro,
                                                                                                         sem_pro,
                                                                                                         input_ppls,
                                                                                                         input_imgs,
                                                                                                         input_num,
                                                                                                         pos_emb_var,
                                                                                                         spa_adj_matrix,
                                                                                                         sem_adj_matrix,
                                                                                                         eval_opt,
                                                                                                         imp_model,
                                                                                                         spa_model,
                                                                                                         sem_model)
        sents = utils.decode_sequence(dataset_val.itow, dataset_val.itod, dataset_val.ltow, dataset_val.itoc,
                                      dataset_val.wtod, seq.data, bn_seq.data, fg_seq.data, opt.vocab_size, opt)
        for k, sent in enumerate(sents):
            entry = {'image_id': img_id[k].item(), 'caption': sent}
            predictions.append(entry)
            if num_show < 20:
                print('image %s: %s' % (entry['image_id'], entry['caption']))
                num_show += 1

        if opt.graph_attention and opt.att_weight_save != '':
            for k in range(len(img_id)):
                save_attention(img_id[k], attention_weights[k], opt.att_weight_save)

    print('Total image to be evaluated %d' % (len(predictions)))
    lang_stats = None
    if opt.language_eval == 1:
        if opt.decode_noc:
            lang_stats = utils.noc_eval(predictions, str(1), opt.val_split, opt)
        else:
            lang_stats = utils.language_eval(opt.dataset, predictions, str(1), opt.val_split, opt)

    print('Saving the predictions')

    # Write validation result into summary
    # if tf is not None:
    #     for k, v in lang_stats.items():
    #         add_summary_value(tf_summary_writer, k, v, iteration)
    #     tf_summary_writer.flush()

    # TODO: change the train process
    # val_result_history[iteration] = {'lang_stats': lang_stats, 'predictions': predictions}
    # if wandb is not None:
    #     wandb.log({k: v for k, v in lang_stats.items()})
    return lang_stats, predictions


def save_attention(img_id, att_weights, path):
    # import pdb
    # pdb.set_trace()
    b = json.dumps(att_weights.cpu().detach().numpy().tolist())
    f2 = open(os.path.join(path, str(img_id.cpu().item()) + '.json'), 'w')
    f2.write(b)
    f2.close()


def fusion_beam_sample(opt, imp_pro, spa_pro, sem_pro, input_ppls, input_imgs, input_num,
                       pos_emb_var, spa_adj_matrix, sem_adj_matrix, eval_opt, imp_model=None, spa_model=None,
                       sem_model=None):
    beam_size = eval_opt.get('beam_size', 3)
    batch_size = input_ppls.data.size(0)
    rois_num = input_ppls.data.size(1)
    nongt_dim = min(rois_num, opt.nongt_dim)
    attention_weights_imp = torch.zeros(batch_size, rois_num, nongt_dim).cuda()
    attention_weights_spa = torch.zeros(batch_size, rois_num, nongt_dim).cuda()
    attention_weights_sem = torch.zeros(batch_size, rois_num, nongt_dim).cuda()
    if imp_model:
        beam_size, fc_feats_imp, conv_feats_imp, pool_feats_imp, p_conv_feats_imp, \
        p_pool_feats_imp, pnt_mask_imp, attention_weights_imp = imp_model.fusion_sample_beam(input_imgs, input_ppls,
                                                                                             input_num, pos_emb_var,
                                                                                             spa_adj_matrix,
                                                                                             sem_adj_matrix,
                                                                                             eval_opt)
    if spa_model:
        beam_size, fc_feats_spa, conv_feats_spa, pool_feats_spa, p_conv_feats_spa, \
        p_pool_feats_spa, pnt_mask_spa, attention_weights_spa = spa_model.fusion_sample_beam(input_imgs, input_ppls,
                                                                                             input_num, pos_emb_var,
                                                                                             spa_adj_matrix,
                                                                                             sem_adj_matrix,
                                                                                             eval_opt)
    if sem_model:
        beam_size, fc_feats_sem, conv_feats_sem, pool_feats_sem, p_conv_feats_sem, \
        p_pool_feats_sem, pnt_mask_sem, attention_weights_sem = sem_model.fusion_sample_beam(input_imgs, input_ppls,
                                                                                             input_num, pos_emb_var,
                                                                                             spa_adj_matrix,
                                                                                             sem_adj_matrix,
                                                                                             eval_opt)
    # import pdb
    # pdb.set_trace()
    vis_offset = (torch.arange(0, beam_size) * rois_num).view(beam_size).type_as(input_ppls.data).long()
    roi_offset = (torch.arange(0, beam_size) * (rois_num + 1)).view(beam_size).type_as(input_ppls.data).long()

    seq = input_ppls.data.new(opt.seq_length, batch_size).zero_().long()
    seqLogprobs = input_ppls.data.new(opt.seq_length, batch_size).float()
    bn_seq = input_ppls.data.new(opt.seq_length, batch_size).zero_().long()
    bnLogprobs = input_ppls.data.new(opt.seq_length, batch_size).float()
    fg_seq = input_ppls.data.new(opt.seq_length, batch_size).zero_().long()
    fgLogprobs = input_ppls.data.new(opt.seq_length, batch_size).float()
    done_beams = [[] for _ in range(batch_size)]
    for k in range(batch_size):
        if imp_model:
            rnn_output_imp, det_prob_imp, state_imp, beam_ppls, beam_pnt_mask_imp, beam_pool_feats_imp, \
            beam_p_pool_feats_imp, beam_fc_feats_imp, beam_conv_feats_imp, beam_p_conv_feats_imp = get_rnn_output(
                imp_model, beam_size, fc_feats_imp, conv_feats_imp,
                pool_feats_imp, p_conv_feats_imp, p_pool_feats_imp,
                input_ppls, rois_num, pnt_mask_imp, k)
            beam_pnt_mask_list_imp = []
            beam_pnt_mask_list_imp.append(beam_pnt_mask_imp)
            beam_att_mask_imp = beam_pnt_mask_imp.clone()
        else:
            rnn_output_imp, det_prob_imp, state_imp = None, None, None
            beam_pnt_mask_list_imp = [torch.from_numpy(np.asarray([]))]
        if spa_model:
            rnn_output_spa, det_prob_spa, state_spa, beam_ppls, beam_pnt_mask_spa, beam_pool_feats_spa, \
            beam_p_pool_feats_spa, beam_fc_feats_spa, beam_conv_feats_spa, beam_p_conv_feats_spa = get_rnn_output(
                spa_model, beam_size, fc_feats_spa, conv_feats_spa,
                pool_feats_spa, p_conv_feats_spa, p_pool_feats_spa,
                input_ppls, rois_num, pnt_mask_spa, k)
            beam_pnt_mask_list_spa = []
            beam_pnt_mask_list_spa.append(beam_pnt_mask_spa)
            beam_att_mask_spa = beam_pnt_mask_spa.clone()
        else:
            rnn_output_spa, det_prob_spa, state_spa = None, None, None
            beam_pnt_mask_list_spa = [torch.from_numpy(np.asarray([]))]
        if sem_model:
            rnn_output_sem, det_prob_sem, state_sem, beam_ppls, beam_pnt_mask_sem, beam_pool_feats_sem, \
            beam_p_pool_feats_sem, beam_fc_feats_sem, beam_conv_feats_sem, beam_p_conv_feats_sem = get_rnn_output(
                sem_model, beam_size, fc_feats_sem, conv_feats_sem,
                pool_feats_sem, p_conv_feats_sem, p_pool_feats_sem,
                input_ppls, rois_num, pnt_mask_sem, k)
            beam_pnt_mask_list_sem = []
            beam_pnt_mask_list_sem.append(beam_pnt_mask_sem)
            beam_att_mask_sem = beam_pnt_mask_sem.clone()
        else:
            rnn_output_sem, det_prob_sem, state_sem = None, None, None
            beam_pnt_mask_list_sem = [torch.from_numpy(np.asarray([]))]

        # beam_att_mask = beam_pnt_mask.clone()

        beam_seq = torch.LongTensor(opt.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(opt.seq_length, beam_size).zero_()
        beam_bn_seq = torch.LongTensor(opt.seq_length, beam_size).zero_()
        beam_bn_seq_logprobs = torch.FloatTensor(opt.seq_length, beam_size).zero_()
        beam_fg_seq = torch.LongTensor(opt.seq_length, beam_size).zero_()
        beam_fg_seq_logprobs = torch.FloatTensor(opt.seq_length, beam_size).zero_()

        beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam
        done_beams_one = []
        # done_beams = [[] for _ in range(batch_size)]
        for t in range(opt.seq_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""

            det_prob = None
            decoded = None

            if imp_model:
                det_prob_imp = F.log_softmax(det_prob_imp, dim=1) * imp_pro
                decoded_imp = F.log_softmax(imp_model.logit(rnn_output_imp), dim=1) * imp_pro
                if det_prob is not None:
                    det_prob = det_prob + det_prob_imp
                    decoded = decoded + decoded_imp
                else:
                    det_prob = det_prob_imp
                    decoded = decoded_imp
            if spa_model:
                det_prob_spa = F.log_softmax(det_prob_spa, dim=1) * spa_pro
                decoded_spa = F.log_softmax(spa_model.logit(rnn_output_spa), dim=1) * spa_pro
                if det_prob is not None:
                    det_prob = det_prob + det_prob_spa
                    decoded = decoded + decoded_spa
                else:
                    det_prob = det_prob_spa
                    decoded = decoded_spa
            if sem_model:
                det_prob_sem = F.log_softmax(det_prob_sem, dim=1) * sem_pro
                decoded_sem = F.log_softmax(sem_model.logit(rnn_output_sem), dim=1) * sem_pro
                if det_prob is not None:
                    det_prob = det_prob + det_prob_sem
                    decoded = decoded + decoded_sem
                else:
                    det_prob = det_prob_sem
                    decoded = decoded_sem

            lambda_v = det_prob[:, 0].contiguous()
            prob_det = det_prob[:, 1:].contiguous()

            decoded = decoded + lambda_v.view(beam_size, 1).expand_as(decoded)
            logprobs = torch.cat([decoded, prob_det], 1)

            logprobsf = logprobs.data.cpu()  # lets go to CPU for more efficiency in indexing operations

            beam_seq, beam_seq_logprobs, \
            beam_logprobs_sum, \
            beam_bn_seq, beam_bn_seq_logprobs, \
            beam_fg_seq, beam_fg_seq_logprobs, \
            rnn_output_imp, rnn_output_spa, rnn_output_sem, \
            state_imp, state_spa, state_sem, \
            beam_pnt_mask_imp_new, beam_pnt_mask_spa_new, beam_pnt_mask_sem_new, \
            candidates_divm = beam_step(logprobsf,
                                        beam_size,
                                        t,
                                        beam_seq,
                                        beam_seq_logprobs,
                                        beam_logprobs_sum,
                                        beam_bn_seq,
                                        beam_bn_seq_logprobs,
                                        beam_fg_seq,
                                        beam_fg_seq_logprobs,
                                        rnn_output_imp,
                                        rnn_output_spa,
                                        rnn_output_sem,
                                        beam_pnt_mask_list_imp[-1].t(),
                                        beam_pnt_mask_list_spa[-1].t(),
                                        beam_pnt_mask_list_sem[-1].t(),
                                        state_imp, state_spa, state_sem)

            it = beam_seq[t].cuda()
            roi_idx = it.clone() - opt.vocab_size - 1  # starting from 0
            roi_mask = roi_idx < 0
            roi_idx_offset = roi_idx + vis_offset
            roi_idx_offset[roi_mask] = 0

            vis_idx = beam_ppls.data[:, :, 4].contiguous().view(-1)[roi_idx_offset].long()
            vis_idx[roi_mask] = 0
            it_new = it.clone()
            it_new[it > opt.vocab_size] = (vis_idx[roi_mask == 0] + opt.vocab_size)

            bn_logprob, fg_logprob = None, None
            if imp_model is not None:
                roi_labels = beam_pool_feats_imp.data.new(beam_size * rois_num).zero_()
                if (roi_mask == 0).sum() > 0: roi_labels[roi_idx_offset[roi_mask == 0]] = 1
                roi_labels = roi_labels.view(beam_size, 1, rois_num)

                bn_logprob_imp, fg_logprob_imp = imp_model.ccr_core(vis_idx, beam_pool_feats_imp,
                                                                    rnn_output_imp.view(beam_size, 1,
                                                                                        imp_model.rnn_size),
                                                                    Variable(roi_labels), beam_size, 1)
                bn_logprob_imp = bn_logprob_imp.view(beam_size, -1) * imp_pro
                fg_logprob_imp = fg_logprob_imp.view(beam_size, -1) * imp_pro
                if bn_logprob is not None:
                    bn_logprob = bn_logprob + bn_logprob_imp
                    fg_logprob = fg_logprob + fg_logprob_imp
                else:
                    bn_logprob = bn_logprob_imp
                    fg_logprob = fg_logprob_imp

            if spa_model is not None:
                roi_labels = beam_pool_feats_spa.data.new(beam_size * rois_num).zero_()
                if (roi_mask == 0).sum() > 0: roi_labels[roi_idx_offset[roi_mask == 0]] = 1
                roi_labels = roi_labels.view(beam_size, 1, rois_num)

                bn_logprob_spa, fg_logprob_spa = spa_model.ccr_core(vis_idx, beam_pool_feats_spa,
                                                                    rnn_output_spa.view(beam_size, 1,
                                                                                        spa_model.rnn_size),
                                                                    Variable(roi_labels), beam_size, 1)
                bn_logprob_spa = bn_logprob_spa.view(beam_size, -1) * spa_pro
                fg_logprob_spa = fg_logprob_spa.view(beam_size, -1) * spa_pro
                if bn_logprob is not None:
                    bn_logprob = bn_logprob + bn_logprob_spa
                    fg_logprob = fg_logprob + fg_logprob_spa
                else:
                    bn_logprob = bn_logprob_spa
                    fg_logprob = fg_logprob_spa

            if sem_model is not None:
                roi_labels = beam_pool_feats_sem.data.new(beam_size * rois_num).zero_()
                if (roi_mask == 0).sum() > 0: roi_labels[roi_idx_offset[roi_mask == 0]] = 1
                roi_labels = roi_labels.view(beam_size, 1, rois_num)

                bn_logprob_sem, fg_logprob_sem = sem_model.ccr_core(vis_idx, beam_pool_feats_sem,
                                                                    rnn_output_sem.view(beam_size, 1,
                                                                                        sem_model.rnn_size),
                                                                    Variable(roi_labels), beam_size, 1)
                bn_logprob_sem = bn_logprob_sem.view(beam_size, -1) * sem_pro
                fg_logprob_sem = fg_logprob_sem.view(beam_size, -1) * sem_pro
                if bn_logprob is not None:
                    bn_logprob = bn_logprob + bn_logprob_sem
                    fg_logprob = fg_logprob + fg_logprob_sem
                else:
                    bn_logprob = bn_logprob_sem
                    fg_logprob = fg_logprob_sem

            slp_bn, it_bn = torch.max(bn_logprob.data, 1)
            slp_fg, it_fg = torch.max(fg_logprob.data, 1)

            it_bn[roi_mask] = 0
            it_fg[roi_mask] = 0

            beam_bn_seq[t] = it_bn
            beam_bn_seq_logprobs[t] = slp_bn

            beam_fg_seq[t] = it_fg
            beam_fg_seq_logprobs[t] = slp_fg

            for vix in range(beam_size):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 0 or t == opt.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(),
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix],
                        'bn_seq': beam_bn_seq[:, vix].clone(),
                        'bn_logps': beam_bn_seq_logprobs[:, vix].clone(),
                        'fg_seq': beam_fg_seq[:, vix].clone(),
                        'fg_logps': beam_fg_seq_logprobs[:, vix].clone(),
                    }

                    done_beams_one.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # updating the mask, and make sure that same object won't happen in the caption
            pnt_idx_offset = roi_idx + roi_offset + 1
            pnt_idx_offset[roi_mask] = 0

            if imp_model:
                beam_pnt_mask_imp = beam_pnt_mask_imp_new.data.clone()

                beam_pnt_mask_imp.view(-1)[pnt_idx_offset] = 1
                beam_pnt_mask_imp.view(-1)[0] = 0
                beam_pnt_mask_list_imp.append(Variable(beam_pnt_mask_imp))

                xt = imp_model.embed(Variable(it_new))
                rnn_output_imp, det_prob_imp, state_imp = imp_model.core(xt, beam_fc_feats_imp, beam_conv_feats_imp,
                                                                         beam_p_conv_feats_imp,
                                                                         beam_pool_feats_imp, beam_p_pool_feats_imp,
                                                                         beam_att_mask_imp, beam_pnt_mask_list_imp[-1],
                                                                         state_imp)
            if spa_model:
                beam_pnt_mask_spa = beam_pnt_mask_spa_new.data.clone()

                beam_pnt_mask_spa.view(-1)[pnt_idx_offset] = 1
                beam_pnt_mask_spa.view(-1)[0] = 0
                beam_pnt_mask_list_spa.append(Variable(beam_pnt_mask_spa))

                xt = spa_model.embed(Variable(it_new))
                rnn_output_spa, det_prob_spa, state_spa = spa_model.core(xt, beam_fc_feats_spa, beam_conv_feats_spa,
                                                                         beam_p_conv_feats_spa,
                                                                         beam_pool_feats_spa, beam_p_pool_feats_spa,
                                                                         beam_att_mask_spa, beam_pnt_mask_list_spa[-1],
                                                                         state_spa)

            if sem_model:
                beam_pnt_mask_sem = beam_pnt_mask_sem_new.data.clone()

                beam_pnt_mask_sem.view(-1)[pnt_idx_offset] = 1
                beam_pnt_mask_sem.view(-1)[0] = 0
                beam_pnt_mask_list_sem.append(Variable(beam_pnt_mask_sem))

                xt = sem_model.embed(Variable(it_new))
                rnn_output_sem, det_prob_sem, state_sem = sem_model.core(xt, beam_fc_feats_sem, beam_conv_feats_sem,
                                                                         beam_p_conv_feats_sem,
                                                                         beam_pool_feats_sem, beam_p_pool_feats_sem,
                                                                         beam_att_mask_sem, beam_pnt_mask_list_sem[-1],
                                                                         state_sem)

        done_beams_one = sorted(done_beams_one, key=lambda x: -x['p'])[:beam_size]
        # import pdb
        # pdb.set_trace()
        done_beams[k] = done_beams_one

        seq[:, k] = done_beams[k][0]['seq'].cuda()  # the first beam has highest cumulative score
        seqLogprobs[:, k] = done_beams[k][0]['logps'].cuda()

        bn_seq[:, k] = done_beams[k][0]['bn_seq'].cuda()
        bnLogprobs[:, k] = done_beams[k][0]['bn_logps'].cuda()

        fg_seq[:, k] = done_beams[k][0]['fg_seq'].cuda()
        fgLogprobs[:, k] = done_beams[k][0]['fg_logps'].cuda()

    if opt.graph_attention:
        attention_weights = attention_weights_imp * imp_pro + attention_weights_spa * spa_pro + attention_weights_sem * sem_pro
    else:
        attention_weights = None
    return seq.t(), bn_seq.t(), fg_seq.t(), seqLogprobs.t(), bnLogprobs.t(), fgLogprobs.t(), attention_weights


def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq,
              beam_bn_seq_logprobs, beam_fg_seq, beam_fg_seq_logprobs, rnn_output_imp, rnn_output_spa,
              rnn_output_sem, beam_pnt_mask_imp, beam_pnt_mask_spa, beam_pnt_mask_sem, state_imp, state_spa, state_sem):
    # INPUTS:
    # logprobsf: probabilities augmented after diversity
    # beam_size: obvious
    # t        : time instant
    # beam_seq : tensor contanining the beams
    # beam_seq_logprobs: tensor contanining the beam logprobs
    # beam_logprobs_sum: tensor contanining joint logprobs
    # OUPUTS:
    # beam_seq : tensor containing the word indices of the decoded captions
    # beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
    # beam_logprobs_sum : joint log-probability of each beam

    ys, ix = torch.sort(logprobsf, 1, True)
    candidates = []
    cols = min(beam_size, ys.size(1))
    rows = beam_size
    if t == 0:
        rows = 1
    for c in range(cols):  # for each column (word, essentially)
        for q in range(rows):  # for each beam expansion
            # compute logprob of expanding beam q with word in (sorted) position c
            local_logprob = ys[q, c]

            candidate_logprob = beam_logprobs_sum[q] + local_logprob
            candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_logprob})

    candidates = sorted(candidates, key=lambda x: -x['p'])

    if state_imp:
        new_state_imp = [_.clone() for _ in state_imp]
    if state_spa:
        new_state_spa = [_.clone() for _ in state_spa]
    if state_sem:
        new_state_sem = [_.clone() for _ in state_sem]

    if rnn_output_imp is not None:
        new_rnn_output_imp = rnn_output_imp.clone()
    if rnn_output_spa is not None:
        new_rnn_output_spa = rnn_output_spa.clone()
    if rnn_output_sem is not None:
        new_rnn_output_sem = rnn_output_sem.clone()
    # new_rnn_output = rnn_output.clone()

    # beam_seq_prev, beam_seq_logprobs_prev
    if t >= 1:
        # we''ll need these as reference when we fork beams around
        beam_seq_prev = beam_seq[:t].clone()
        beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()

        beam_bn_seq_prev = beam_bn_seq[:t].clone()
        beam_bn_seq_logprobs_prev = beam_bn_seq_logprobs[:t].clone()

        beam_fg_seq_prev = beam_fg_seq[:t].clone()
        beam_fg_seq_logprobs_prev = beam_fg_seq_logprobs[:t].clone()

        if len(beam_pnt_mask_imp) > 0:
            beam_pnt_mask_prev_imp = beam_pnt_mask_imp.clone()
            beam_pnt_mask_imp = beam_pnt_mask_imp.clone()
        if len(beam_pnt_mask_spa) > 0:
            beam_pnt_mask_prev_spa = beam_pnt_mask_spa.clone()
            beam_pnt_mask_spa = beam_pnt_mask_spa.clone()
        if len(beam_pnt_mask_sem) > 0:
            beam_pnt_mask_prev_sem = beam_pnt_mask_sem.clone()
            beam_pnt_mask_sem = beam_pnt_mask_sem.clone()
        # beam_pnt_mask_prev = beam_pnt_mask.clone()
        # beam_pnt_mask = beam_pnt_mask.clone()

    for vix in range(beam_size):
        v = candidates[vix]
        # fork beam index q into index vix
        if t >= 1:
            beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
            beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
            beam_bn_seq[:t, vix] = beam_bn_seq_prev[:, v['q']]
            beam_bn_seq_logprobs[:t, vix] = beam_bn_seq_logprobs_prev[:, v['q']]
            beam_fg_seq[:t, vix] = beam_fg_seq_prev[:, v['q']]
            beam_fg_seq_logprobs[:t, vix] = beam_fg_seq_logprobs_prev[:, v['q']]

            if len(beam_pnt_mask_imp) > 0:
                beam_pnt_mask_imp[:, vix] = beam_pnt_mask_prev_imp[:, v['q']]
            if len(beam_pnt_mask_spa) > 0:
                beam_pnt_mask_spa[:, vix] = beam_pnt_mask_prev_spa[:, v['q']]
            if len(beam_pnt_mask_sem) > 0:
                beam_pnt_mask_sem[:, vix] = beam_pnt_mask_prev_sem[:, v['q']]
            # beam_pnt_mask[:, vix] = beam_pnt_mask_prev[:, v['q']]

        # rearrange recurrent states
        if state_imp:
            for state_ix in range(len(new_state_imp)):
                #  copy over state in previous beam q to new beam at vix
                new_state_imp[state_ix][:, vix] = state_imp[state_ix][:, v['q']]  # dimension one is time step

            new_rnn_output_imp[vix] = rnn_output_imp[v['q']]  # dimension one is time step
        if state_spa:
            for state_ix in range(len(new_state_spa)):
                #  copy over state in previous beam q to new beam at vix
                new_state_spa[state_ix][:, vix] = state_spa[state_ix][:, v['q']]  # dimension one is time step

            new_rnn_output_spa[vix] = rnn_output_spa[v['q']]  # dimension one is time step
        if state_sem:
            for state_ix in range(len(new_state_sem)):
                #  copy over state in previous beam q to new beam at vix
                new_state_sem[state_ix][:, vix] = state_sem[state_ix][:, v['q']]  # dimension one is time step

            new_rnn_output_sem[vix] = rnn_output_sem[v['q']]  # dimension one is time step

        # append new end terminal at the end of this beam
        beam_seq[t, vix] = v['c']  # c'th word is the continuation
        beam_seq_logprobs[t, vix] = v['r']  # the raw logprob here
        beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam

    if state_imp:
        state_imp = new_state_imp
        rnn_output_imp = new_rnn_output_imp
    if state_spa:
        state_spa = new_state_spa
        rnn_output_spa = new_rnn_output_spa
    if state_sem:
        state_sem = new_state_sem
        rnn_output_sem = new_rnn_output_sem

    return beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_bn_seq, beam_bn_seq_logprobs, \
           beam_fg_seq, beam_fg_seq_logprobs, rnn_output_imp, rnn_output_spa, rnn_output_sem, \
           state_imp, state_spa, state_sem, beam_pnt_mask_imp.t(), beam_pnt_mask_spa.t(), beam_pnt_mask_sem.t(), \
           candidates
