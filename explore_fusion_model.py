from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import opts

from evaluation import eval_relationNBT
from misc import utils
import yaml

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

#  CUDA_VISIBLE_DEVICES=2 python explore_fusion_model.py --path_opt cfgs/normal_coco_res101.yml --batch_size 100 --cuda True --num_workers 1 --beam_size 3 --sem_model --spa_model --imp_model --sem_pro 0.3 --spa_pro 0.3 --imp_pro 0.4
if __name__ == '__main__':
    opt = opts.parse_opt()
    if opt.path_opt is not None:
        with open(opt.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
        utils.update_values(options_yaml, vars(opt))
    print(opt)
    # sys.stdout = Logger('save/explore_fustion_model_info.log', sys.stdout)
    # sys.stderr = Logger('save/explore_fustion_model_info.log', sys.stderr)
    j = 8
    for i in range(1, 9):
        opt.spa_pro = round(i * 0.1, 1)
        opt.sem_pro = round(j * 0.1, 1)
        opt.imp_pro = round(1 - opt.spa_pro - opt.sem_pro, 1)
        print("-----------------new paramters----------------------")
        print("spa_pro: {}, sem_pro: {}, imp_pro: {}".format(opt.spa_pro, opt.sem_pro, opt.imp_pro))
        eval_relationNBT(opt)
        j -= 1
    # for i in range(1, 9):
    #     opt.spa_pro = round(i * 0.1, 1)
    #     for j in range(1, 9 - i):
    #         opt.sem_pro = round(j * 0.1, 1)
    #         opt.imp_pro = round(1 - opt.spa_pro - opt.sem_pro, 1)
    #         print("-----------------new paramters----------------------")
    #         print("spa_pro: {}, sem_pro: {}, imp_pro: {}".format(opt.spa_pro, opt.sem_pro, opt.imp_pro))
    #         eval_relationNBT(opt)