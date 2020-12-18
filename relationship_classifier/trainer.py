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


def train(cfgs):
    # dataset

    # train_dataset = BoxesDataset(os.path.join(cfgs.data_path, 'train_vg.json'), cfgs.boxes_path, cfgs.union_path)
    # train_loader = data.DataLoader(dataset=train_dataset, batch_size=cfgs.batch_size, num_workers=1, shuffle=False)
    #
    # val_dataset = BoxesDataset(os.path.join(cfgs.data_path, 'val_vg.json'), cfgs.boxes_path, cfgs.union_path)
    # val_loader = data.DataLoader(dataset=val_dataset, batch_size=cfgs.batch_size, shuffle=False)
    train_dataset = BoxesDataset(os.path.join(cfgs.data_path, 'train_union_info.json'),
                                 os.path.join(cfgs.data_path, 'train_union_features.json'),
                                 os.path.join(cfgs.data_path, 'train_boxes_features.json'))
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=cfgs.batch_size, num_workers=4, shuffle=True)

    val_dataset = BoxesDataset(os.path.join(cfgs.data_path, 'val_union_info.json'),
                                 os.path.join(cfgs.data_path, 'val_union_features.json'),
                                 os.path.join(cfgs.data_path, 'val_boxes_features.json'))
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=cfgs.batch_size, num_workers=1, shuffle=False)

    # Model
    model = Classifier(0.5)
    if cfgs.resume:
        checkpoint = torch.load(cfgs.checkpoint + 'checkpoint_final.pkl')
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        learning_rate = checkpoint['learning_rate']
        train_loss_epoch = checkpoint['train_loss_epoch']
        train_acc_epoch = checkpoint['train_acc_epoch']
        test_acc_epoch = checkpoint['test_acc_epoch']
    else:
        epoch = 0
        learning_rate = cfgs.learning_rate
        train_loss_epoch = []
        train_acc_epoch = []
        test_acc_epoch = []

    if cfgs.mGPUs:
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model.cuda()

    if wandb is not None:
        wandb.watch(model)
    # optimizer
    criterion = torch.nn.CrossEntropyLoss()

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            params += [{'params': [value], 'lr': learning_rate}]
                            # 'weight_decay': opt.weight_decay, 'betas': (opt.optim_alpha, opt.optim_beta)}]


    # optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
    optimizer = optim.Adam(params)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    best_acc = -1.0

    # for epoch in range(cfgs.max_epochs):
    while epoch < cfgs.max_epochs:
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        count = 0
        data_len = 0
        # update learning rate
        if 0 < epoch < 4:
            for group in optimizer.param_groups:
                group['lr'] = group['lr'] + cfgs.learning_rate
                learning_rate = group['lr']
        if 4 <= epoch < 20 and (epoch + 1) % 2 == 0:
            for group in optimizer.param_groups:
                group['lr'] = group['lr'] / 2
                learning_rate = group['lr']

        # start_time = time.time()
        # end_time = time.time()
        progress_bar = tqdm(train_loader, desc='|Train Epoch {}'.format(epoch), leave=False)
        for i, batch in enumerate(progress_bar):
            # end_time = time.time()
            # print('Done (t={:0.2f}s)'.format(end_time - start_time))
            count += 1
            obj_feat, sub_feat, union_feat, labels = batch
            obj_feat, sub_feat, union_feat, labels = obj_feat.cuda(), sub_feat.cuda(), union_feat.cuda(), labels.cuda()

            length = labels.size(0)
            data_len += length
            optimizer.zero_grad()
            outputs = model(obj_feat, sub_feat, union_feat)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # pring statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            accuracy = torch.sum(predicted == labels).item()
            train_acc += accuracy

            train_loss_log = loss.item()
            train_acc_log = float(accuracy) / length
            info_log = {
                'train_loss': loss.item(),
                'train_accuracy': accuracy / length,
            }

            progress_bar.set_postfix(info_log,  refresh=True)
            if wandb is not None:
                wandb.log({
                    'train_loss_it': train_loss_log,
                    'train_accuracy_it': train_acc_log
                })
            # start_time = time.time()

        loss_aveg = float(train_loss) / count
        acc_aveg = float(train_acc) / data_len
        print('Train Epoch: {}, train_loss: {}, train_accuracy: {}, lr: {}.'.format(epoch, loss_aveg, acc_aveg, learning_rate))
        train_loss_epoch.append(loss_aveg)
        train_acc_epoch.append(acc_aveg)
        if wandb is not None:
            wandb.log({
                'train_loss_epoch': loss_aveg,
                'train_acc_epoch': acc_aveg
            })

        # scheduler.step()
        # caculate the test accuracy
        model.eval()
        # if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            test_total = 0
            test_correct = 0
            process_bar_test = tqdm(val_loader, desc='|Test Epoch {}'.format(epoch), leave=False)
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
                test_acc_log = float(correct / labels.size(0))
                if wandb is not None:
                    wandb.log({
                        'test_acc_it': test_acc_log
                    })

            test_acc_aveg = float(test_correct) / test_total
            if wandb is not None:
                wandb.log({
                    'test_acc_epoch': test_acc_aveg
                })
            if test_acc_aveg > best_acc:
                best_acc = test_acc_aveg
                if cfgs.mGPUs:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'learning_rate': cfgs.learning_rate,
                        'loss': loss_aveg,
                        'accuracy': acc_aveg,
                        'test_accuracy': test_acc_aveg
                    }, cfgs.checkpoint + 'checkpoint_best.pkl')
                else:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'learning_rate': cfgs.learning_rate,
                        'loss': loss_aveg,
                        'accuracy': acc_aveg,
                        'loss_val': test_acc_aveg
                    }, cfgs.checkpoint + 'checkpoint_best.pkl')
            print('Epoch: {}, Accuracy of the model on testset: {}'.format(epoch, test_acc_aveg))
            test_acc_epoch.append(test_acc_aveg)

        epoch += 1

    if epoch == cfgs.max_epochs:
        if cfgs.mGPUs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'learning_rate': cfgs.learning_rate,
                'train_loss_epoch': train_loss_epoch,
                'train_acc_epoch': train_acc_epoch,
                'test_acc_epoch': test_acc_epoch
            }, cfgs.checkpoint + 'checkpoint_final.pkl')
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': cfgs.learning_rate,
                'train_loss_epoch': train_loss_epoch,
                'train_acc_epoch': train_acc_epoch,
                'test_acc_epoch': test_acc_epoch
            }, cfgs.checkpoint + 'checkpoint_final.pkl')


if __name__ == '__main__':
    wandb.init(project="relationship_classifier")
    opt = opts.parse_opt()
    sys.stdout = utils.Logger(opt.output_dir + 'classifier/info_weighNorm.log', sys.stdout)
    sys.stderr = utils.Logger(opt.output_dir + 'classifier/error_weighNorm.log', sys.stderr)
    train(opt)
