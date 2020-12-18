import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Classifier(nn.Module):

    def __init__(self, dropout=0.0):
        super(Classifier, self).__init__()

        self.fc_obj = weight_norm(nn.Linear(2048, 512), name='weight')
        self.obj_drop = torch.nn.Dropout(dropout)
        self.fc_sub = weight_norm(nn.Linear(2048, 512), name='weight')
        self.sub_drop = torch.nn.Dropout(dropout)
        self.fc_union = weight_norm(nn.Linear(2048, 512), name='weight')
        self.union_drop = torch.nn.Dropout(dropout)

        self.fc_classification = nn.Linear(512 * 3, 16)
        self.dropout = dropout
        # self.soft_max = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, obj, sub, union):

        if self.dropout != 0:
            obj_feature = F.relu(self.obj_drop(self.fc_obj(obj)))
            sub_feature = F.relu(self.sub_drop(self.fc_sub(sub)))
            union_feature = F.relu(self.union_drop(self.fc_union(union)))
        else:
            obj_feature = F.relu(self.fc_obj(obj))
            sub_feature = F.relu(self.fc_sub(sub))
            union_feature = F.relu(self.fc_union(union))

        feature = torch.cat((obj_feature, sub_feature, union_feature), 1)

        x = self.fc_classification(feature)

        return x

    def predictor(self, obj, sub, union):

        obj_feature = F.relu(self.fc_obj(obj))
        sub_feature = F.relu(self.fc_sub(sub))
        union_feature = F.relu(self.fc_union(union))

        feature = torch.cat((obj_feature, sub_feature, union_feature), 1)

        x = self.fc_classification(feature)

        output = F.softmax(x, dim=1)

        return output

    # initialize weights
    def _init_weights(self):
        for _m in self.modules():
            if isinstance(_m, nn.Linear):
                nn.init.xavier_uniform_(_m.weight.data)
                _m.bias.data.fill_(0.1)
