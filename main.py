# -*- coding: UTF-8 -*-
from builtins import print

import shutil
import time
import argparse

from datetime import datetime

from tensorboardX import SummaryWriter
from torch.autograd import Variable

from os.path import isfile, join, isdir

from data_process.gauss import gaussian_label_function

from logs.logger import *
from data_process.dateFCppOp import Pair
from data_process.utils import *
import os
from losses.totalloss import TotalLoss
from lr import *

from network.R3DCppOp3D import SiamR3D, R3DNet, SpatioTemporalResBlock




parser = argparse.ArgumentParser(description='Training Siam3D in Pytorch ')
parser.add_argument('--numEpochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=4, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--lr', default= 0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--batch-size', default=11, type=int, metavar='N', help='mini-batch size (default: 48)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save_dir', default='modelPath', type=str,help='directory for saving')
parser.add_argument('--dataset', default='datasetPath', type=str,help='path to original datasets')
parser.add_argument('--log_dir', default='boardPath', help='TensorBoard log dir')

best_auc = 0
logger = get_logger('logPath')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


args = parser.parse_args()


# parameters configuration(dataset)
class train_config(object):
    exemplarSize = 127
    instanceSize = 255
    scoreSize = 17
    context = 0.5
    rPos = 16
    rNeg = 0
    totalStride = 8
    ignoreLabel = -100
    lr = 1e-5  # 学习率


class track_config(object):
    numScale = 3
    scaleStep = 1.0375
    scalePenalty = 0.9745
    scaleLR = 0.59
    responseUp = 16
    wInfluence = 0.176
    z_lr = 0.1
    scale_min = 0.2
    scale_max = 5

    exemplarSize = 127
    instanceSize = 255
    scoreSize = 17
    totalStride = 8
    contextAmount = 0.5
    final_sz = responseUp * (scoreSize - 1) + 1


def save_checkpoint(state, tr):
    if tr:
         save_file='model.pth'
    else:
        save_file='model.pth'
    torch.save(state, save_file)


def train(train_loader, val_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    logger.info(args)
   
    model.train()
    model=model.to(device)


    num_per_epoch = len(train_loader) // args.numEpochs // args.batch_size
    start_epoch = args.start_epoch
    epoch = epoch
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    
    lr_scheduler = ListLR(
        LinearLR(start_lr=1e-5, end_lr=1e-1, max_epoch=1, max_iter=15000),
        LinearLR(start_lr=1e-1, end_lr=1e-5, max_epoch=20, max_iter=15000))

    trainiou=0

    for e in range(epoch):
        train_iou_list=[]
        train_loss_list=[]
     
        for iter, (template, search, cls_labels, ctr_labels, box_labels,gauss_label) in enumerate(train_loader):
           
            template = Variable(template).to(device)
            search = Variable(search).to(device)
            cls_labels = Variable(cls_labels).to(device)
            ctr_labels = Variable(ctr_labels).to(device)
            box_labels = Variable(box_labels).to(device)
           

            search = search.float()
            template = template.float()
            
            gauss_label= Variable(gauss_label).to(device)
            gauss_label=gauss_label.float()

            adjust_learning_rate(lr_scheduler, optimizer, e, iter)

            if iter % num_per_epoch == 0 and iter != 0:
                for idx, pg in enumerate(optimizer.param_groups):
                    logger.info("epoch {} lr {}".format(e, pg['lr']))
                    tb_writer.add_scalar('lr/group%d' % (idx + 1), pg['lr'], e)


            # data_time = time.time() - end
            toc = time.time()
            
            xcorr_clss, xcorr_ctrs, xcorr_boxs = model(search, template,gauss_label)
            

            cls_label = cls_labels.reshape(-1, 17 * 17, 1)
            ctr_label = ctr_labels.reshape(-1, 17 * 17, 1)
            box_label = box_labels.reshape(-1, 17 * 17, 4)

            xcorr_cls = xcorr_clss.reshape(-1, 17 * 17, 1)
            xcorr_ctr = xcorr_ctrs.reshape(-1, 17 * 17, 1)
            xcorr_box = xcorr_boxs.reshape(-1, 17 * 17, 4)
         


            loss, extra = criterion(xcorr_cls, xcorr_ctr, xcorr_box, cls_label, ctr_label, box_label)

            # record loss
            losses.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.data.cpu().numpy())
            train_loss = np.mean(train_loss_list)

            train_iou_list.append(extra.data.cpu().numpy())
            train_iou = np.mean(train_iou_list)

            tb_writer.add_scalar('loss/train', train_loss, e)
            tb_writer.add_scalar('iou/train', train_iou, e)

            
          

            
            batch_time = time.time() - toc
            if (iter + 1) % args.print_freq == 0:
                print(
                    "iou:", extra,
                    '[{0}Epoch]: [{1}][{2}/{3}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        e, epoch, iter, len(train_loader), loss=losses))

      
        if trainiou<train_iou:
            trainiou=train_iou
  
        model.eval()
        valiou = val(val_loader, model, criterion, epoch=e)
       
        model.train()

        if e == 0:
            
            saveiou = valiou
           
            # save train model
            if not isdir(args.save_dir):
                os.makedirs(args.save_dir)
            save_checkpoint({
                'epoch': e + 1,
                'state_dict': model.state_dict(),
                'saveiou': saveiou,
                'optimizer': optimizer.state_dict(),
            }, epoch)
            print(e,'trainiou:',trainiou,'valiou:', valiou)
        else:
            
            if saveiou < valiou:
                tr=True
                saveiou = valiou
                if not isdir(args.save_dir):
                    os.makedirs(args.save_dir)
                save_checkpoint({
                    'epoch': e + 1,
                    'state_dict': model.state_dict(),
                    'saveiou': saveiou,
                    'optimizer': optimizer.state_dict(),
                }, tr)
            else:
                tr=False
                save_checkpoint({
                    'epoch': e + 1,
                    'state_dict': model.state_dict(),
                    'trainiou': trainiou,
                    'optimizer': optimizer.state_dict(),
                }, tr)
            print(e,'trainiou:',trainiou,'saveiou:', saveiou)


def val(val_loader, model, criterion, epoch):
    logger.info(args)
    val_loss_list = []
    val_iou_list=[]

    for iter, (template, search, cls_labels, ctr_labels, box_labels,gauss_label) in enumerate(val_loader):

            template = Variable(template).to(device)
            search = Variable(search).to(device)
            cls_labels = Variable(cls_labels).to(device)
            ctr_labels = Variable(ctr_labels).to(device)
            box_labels = Variable(box_labels).to(device)
         

            search = search.float()
            template = template.float()
          
            gauss_label= Variable(gauss_label).to(device)
            gauss_label=gauss_label.float()

            with torch.no_grad():      
                xcorr_clss, xcorr_ctrs, xcorr_boxs = model(search, template,gauss_label)


            cls_label = cls_labels.reshape(-1, 17 * 17, 1)
            ctr_label = ctr_labels.reshape(-1, 17 * 17, 1)
            box_label = box_labels.reshape(-1, 17 * 17, 4)

            xcorr_cls = xcorr_clss.reshape(-1, 17 * 17, 1)
            xcorr_ctr = xcorr_ctrs.reshape(-1, 17 * 17, 1)
            xcorr_box = xcorr_boxs.reshape(-1, 17 * 17, 4)
         


            loss, extra = criterion(xcorr_cls, xcorr_ctr, xcorr_box, cls_label, ctr_label, box_label)

            val_loss_list.append(loss.data.cpu().numpy())
            val_loss = np.mean(val_loss_list)

            val_iou_list.append(extra.data.cpu().numpy())
            val_iou = np.mean(val_iou_list)

            tb_index = epoch
            # record loss
            tb_writer.add_scalar('loss/val', val_loss, tb_index)
            tb_writer.add_scalar('iou/val', val_iou, tb_index)

            end = time.time()
            if (iter + 1) % args.print_freq == 0:
                print("VAL---", "iou:", extra,
                    '[{0}Epoch]:[{1}/{2}]\t'
                    'Loss ({loss:.4f})\t'.format(
                        epoch, iter, len(val_loader), loss=loss))

    return val_iou


def main():
    global saveiou, cur_lr, tb_writer

    if args.log_dir:
        time_str = datetime.now().__str__().replace(":", '-').replace(' ', '_')
        cur_log_dir = os.path.join(args.log_dir, time_str)
        os.mkdir(cur_log_dir)
        tb_writer = SummaryWriter(cur_log_dir)
    else:
        tb_writer = Dummy()
    # dataset(train, val)参数设置

    root_dir = 'datasetPath'
   
    # 训练数据对
    pair_train = Pair(root_dir, subset='train', config=train_config(), pairs_per_video=30, rand_choice=True,
                      frame_range=100)
    train_loader = torch.utils.data.DataLoader(pair_train, batch_size=args.batch_size, shuffle=True,
                                               num_workers=0, pin_memory=False, drop_last=True)
    # 验证数据对
    pair_val = Pair(root_dir, subset='val', config=train_config(), pairs_per_video=30, rand_choice=True,
                    frame_range=100)
    val_loader = torch.utils.data.DataLoader(pair_val, batch_size=args.batch_size, shuffle=True,
                                             num_workers=0, pin_memory=False, drop_last=True)
    logger.info('build dataset done')
    net = R3DNet((2, 2, 2, 2), block_type=SpatioTemporalResBlock)
 
    model = SiamR3D(branch=net)
    
    criterion = TotalLoss()

    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # define optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    train(train_loader, val_loader, model, criterion,  optimizer, epoch=20)
    



if __name__ == '__main__':
    main()

