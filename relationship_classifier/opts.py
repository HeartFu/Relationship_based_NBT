import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # # Data input settings
    # parser.add_argument('--data_path', type=str, default='/home/fanfu/newdisk/dataset/VisualGenome/feature/',
    #                     help='features path')
    parser.add_argument('--data_path', type=str, default='/import/nobackup_mmv_ioannisp/tx301/vg_feature/vg_dataset/features_2/',
                        help='features path')
    # parser.add_argument('--boxes_path', type=str, default='/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/data/boxes_feature/',
    #                     help='The path of bounding boxes feature')
    # parser.add_argument('--union_path', type=str, default='/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/data/union_boxes/',
    #                     help='The path of union boxes feature')
    # parser.add_argument('--data_path', type=str,
    #                     default='',
    #                     help='')
    parser.add_argument('--boxes_path', type=str,
                        default='/home/fanfu/newdisk/pytorch-bottom-up-attention/py-bottom-up-attention/demo/relationship_classifier/data/boxes_feature/',
                        help='The path of bounding boxes feature')
    parser.add_argument('--union_path', type=str,
                        default='/home/fanfu/newdisk/pytorch-bottom-up-attention/py-bottom-up-attention/demo/relationship_classifier/data/union_boxes',
                        help='The path of union boxes feature')

    parser.add_argument('--train_proportion', type=float, default=0.8,
                        help='The path of union boxes feature')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Initial Learning Rate')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='minibatch size')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--mGPUs', type=bool, default=True,
                        help='multi GPU run')

    # Log
    parser.add_argument('--output_dir', type=str, default='log/',
                        help='dir which save the log file')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/weight_norm/',
                        help='Checkpoint path')
    parser.add_argument('--resume', type=bool, default=False,
                        help='whether recovery training process')


    args = parser.parse_args()

    return args
