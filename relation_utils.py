import math

from torch.autograd import Variable
import torch
import numpy as np


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def build_spatial_graph(bbox, image_h, image_w, label_num=11):
    """
    Building the spatial graph for spatial relation module
    Args:
        bbox: [num_boxes, 4]
        label_num: default 11 categories

    Returns:
        adj_matrix: [num_boxes, num_boxes, label_num]
    """
    num_box = bbox.shape[0]
    adj_matrix = np.zeros((num_box, num_box))
    xmin, ymin, xmax, ymax = np.split(bbox, 4, axis=1) # [num_boxes, 1]
    bbox_width = xmax - xmin + 1
    bbox_height = ymax - ymin + 1
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)

    image_diag = math.sqrt(image_h**2 + image_w**2)
    for i in range(num_box):
        bbA = bbox[i]
        if sum(bbA) == 0:
            continue
        adj_matrix[i, i] = 12
        for j in range(i + 1, num_box):
            bbB = bbox[j]
            if sum(bbB) == 0:
                continue
            # class 1: inside (j inside i)
            if xmin[i] < xmin[j] and xmax[i] > xmax[j] and \
                    ymin[i] < ymin[j] and ymax[i] > ymax[j]:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 2
            # class 2: cover (j covers i)
            elif (xmin[j] < xmin[i] and xmax[j] > xmax[i] and
                  ymin[j] < ymin[i] and ymax[j] > ymax[i]):
                adj_matrix[i, j] = 2
                adj_matrix[j, i] = 1
            else:
                ioU = bb_intersection_over_union(bbA, bbB)
                # class 3: i and j overlap
                if ioU >= 0.5:
                    adj_matrix[i, j] = 3
                    adj_matrix[j, i] = 3
                else:
                    y_diff = center_y[i] - center_y[j]
                    x_diff = center_x[i] - center_x[j]
                    diag = math.sqrt((y_diff) ** 2 + (x_diff) ** 2)
                    if diag < 0.5 * image_diag: # make sure that the two objects has the position relationship
                        sin_ij = y_diff / diag
                        cos_ij = x_diff / diag
                        if sin_ij >= 0 and cos_ij >= 0:
                            label_i = np.arcsin(sin_ij)
                            label_j = 2 * math.pi - label_i
                        elif sin_ij < 0 and cos_ij >= 0:
                            label_i = np.arcsin(sin_ij) + 2 * math.pi
                            label_j = label_i - math.pi
                        elif sin_ij >= 0 and cos_ij < 0:
                            label_i = np.arccos(cos_ij)
                            label_j = 2 * math.pi - label_i
                        else:
                            label_i = -np.arccos(sin_ij) + 2 * math.pi
                            label_j = label_i - math.pi
                        adj_matrix[i, j] = int(np.ceil(label_i / (math.pi / 4))) + 3
                        adj_matrix[j, i] = int(np.ceil(label_j / (math.pi / 4))) + 3
    return adj_matrix


def extract_implict_position(rois, nongt_dim, pos_emb_dim):
    bb = rois[:, :, 1:]

    pos_mat = torch_extract_position_matrix(bb, nongt_dim=nongt_dim)
    pos_emb = torch_extract_position_embedding(
        pos_mat, feat_dim=pos_emb_dim)
    # import pdb
    # pdb.set_trace()
    # pos_emb_var = Variable(pos_emb).cuda()

    # return pos_emb_var
    return pos_emb


def torch_extract_position_matrix(bbox, nongt_dim=36):
    """ Extract position matrix

    Args:
        bbox: [batch_size, num_boxes, 4]

    Returns:
        position_matrix: [batch_size, num_boxes, nongt_dim, 4]
    """

    xmin, ymin, xmax, ymax = torch.split(bbox, 1, dim=-1)
    # [batch_size,num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    # [batch_size,num_boxes, num_boxes]
    delta_x = center_x - torch.transpose(center_x, 1, 2)
    delta_x = torch.div(delta_x, bbox_width)
    delta_x = torch.abs(delta_x)
    threshold = 1e-3
    delta_x[delta_x < threshold] = threshold
    delta_x = torch.log(delta_x)

    delta_y = center_y - torch.transpose(center_y, 1, 2)
    delta_y = torch.div(delta_y, bbox_height)
    delta_y = torch.abs(delta_y)
    delta_y[delta_y < threshold] = threshold
    delta_y = torch.log(delta_y)

    delta_width = torch.div(bbox_width, torch.transpose(bbox_width, 1, 2))
    delta_width = torch.log(delta_width)

    delta_height = torch.div(bbox_height, torch.transpose(bbox_height, 1, 2))
    delta_height = torch.log(delta_height)

    concat_list = [delta_x, delta_y, delta_width, delta_height]
    for idx, sym in enumerate(concat_list):
        sym = sym[:, :nongt_dim]
        concat_list[idx] = torch.unsqueeze(sym, dim=3)
    position_matrix = torch.cat(concat_list, 3)
    return position_matrix


def torch_extract_position_embedding(position_mat, feat_dim, wave_length=1000):
    # position_mat, [batch_size,num_rois, nongt_dim, 4]
    feat_range = torch.arange(0, feat_dim / 8)
    dim_mat = torch.pow(torch.ones((1,)) * wave_length,
                        (8. / feat_dim) * feat_range)

    dim_mat = dim_mat.view(1, 1, 1, -1)
    position_mat = torch.unsqueeze(100.0 * position_mat, dim=4)
    div_mat = torch.div(position_mat, dim_mat)
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    # embedding, [batch_size,num_rois, nongt_dim, 4, feat_dim/4]
    embedding = torch.cat([sin_mat, cos_mat], -1)
    # embedding, [batch_size,num_rois, nongt_dim, feat_dim]
    embedding = embedding.view(embedding.shape[0], embedding.shape[1],
                               embedding.shape[2], feat_dim)
    return embedding


def torch_broadcast_adj_matrix(adj_matrix, label_num=11):
    """ broudcast spatial relation graph

    Args:
        adj_matrix: [batch_size,num_boxes, num_boxes]

    Returns:
        result: [batch_size,num_boxes, num_boxes, label_num]
    """
    result = []
    for i in range(1, label_num+1):
        index = torch.nonzero((adj_matrix == i).view(-1).data).squeeze()
        curr_result = torch.zeros(
            adj_matrix.shape[0], adj_matrix.shape[1], adj_matrix.shape[2])
        curr_result = curr_result.view(-1)
        curr_result[index] += 1
        result.append(curr_result.view(
            (adj_matrix.shape[0], adj_matrix.shape[1],
             adj_matrix.shape[2], 1)))
    result = torch.cat(result, dim=3)
    return result


def prepare_graph_variables(relation_type, bb, sem_adj_matrix, spa_adj_matrix,
                            nongt_dim, pos_emb_dim, spa_label_num,
                            sem_label_num, eval_opt=None):
    if eval_opt is None:
        eval_opt = {
            'imp_model': False,
            'spa_model': False,
            'sem_model': False,
            'graph_att': True,
        }
    pos_emb_var, sem_adj_matrix_var, spa_adj_matrix_var = None, None, None
    if relation_type == 'spatial' or eval_opt['spa_model']:
        assert spa_adj_matrix.dim() > 2, "Found spa_adj_matrix of wrong shape"
        # spa_adj_matrix = spa_adj_matrix[:, :num_objects, :num_objects]
        if eval_opt['graph_att']:
            spa_adj_matrix_bc = torch_broadcast_adj_matrix(
                spa_adj_matrix, label_num=spa_label_num)
            spa_adj_matrix_var = Variable(spa_adj_matrix_bc).cuda()
        else:
            spa_adj_matrix_var = Variable(spa_adj_matrix).cuda()
    if (relation_type == 'implicit' or eval_opt['imp_model']):
        pos_mat = torch_extract_position_matrix(bb, nongt_dim=nongt_dim)
        pos_emb = torch_extract_position_embedding(
            pos_mat, feat_dim=pos_emb_dim)
        pos_emb_var = Variable(pos_emb).cuda()
    if relation_type == 'semantic' or eval_opt['sem_model']:
        assert sem_adj_matrix.dim() > 2, "Found sem_adj_matrix of wrong shape"
        if eval_opt['graph_att']:
            sem_adj_matrix_bc = torch_broadcast_adj_matrix(
                sem_adj_matrix, label_num=sem_label_num)
            sem_adj_matrix_var = Variable(sem_adj_matrix_bc).cuda()
        else:
            sem_adj_matrix_var = Variable(sem_adj_matrix).cuda()
    return pos_emb_var, spa_adj_matrix_var, sem_adj_matrix_var