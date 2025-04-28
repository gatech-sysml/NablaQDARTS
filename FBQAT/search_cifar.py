#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 06:26:11 2021

@author: root
"""

import argparse
import os
import random
import shutil
import time
import warnings

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "DARTS_with_Post_Training_Quantization"))

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models as models
from models.quant_darts import turn_into_mixed_quant_aware_model, remove_all_useless_layers
from models.mixdarts import DARTS_CIFAR
from models import quant_module as qm
# from DARTS_with_Post_Training_Quantization.cnn.model import NetworkCIFAR
from DARTS_with_Post_Training_Quantization.cnn import genotypes



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')

parser.add_argument('--data', type=str, default='data', help='location of the data corpus')

parser.add_argument('-a', '--arch', metavar='ARCH', default='DARTS',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: DARTS)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--step-epoch', default=30, type=int, metavar='N',
                    help='number of epochs to decay learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lra', '--learning-rate-alpha', default=0.01, type=float,
                    metavar='LR', help='initial alpha learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--complexity-decay', '--cd', default=0, type=float,
                    metavar='W', help='complexity decay (default: 1e-4)')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def data_transforms_cifar10():
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])


  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def main():
    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch=='DARTS':
        model = DARTS_CIFAR(drop_path_prob = args.drop_path_prob, conv_func = qm.MixActivConv2d, C=36, num_classes=10, layers = 20, auxiliary=True, genotype=genotypes.DARTS)

        # Do backward pass so we can identify which layers do not have a gradient and remove them
        model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()

        out, _ = model(torch.ones((5, 3, 32, 32)).cuda())
        ((abs(out)).sum() + (abs(_).sum())).backward()
        remove_all_useless_layers(model)
        optimizer.zero_grad()    
        model = turn_into_mixed_quant_aware_model(model, wbits = [2,4], abits = [2,4])
    
    else:
        model = models.__dict__[args.arch]()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if 'alex' in args.arch or 'vgg' in args.arch:
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # group model/architecture parameters
    params, alpha_params = [], []
    for name, param in model.named_parameters():
        if 'alpha' in name:
            alpha_params += [param]
        else:
            params += [param]
    optimizer = torch.optim.SGD(params, args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    arch_optimizer = torch.optim.SGD(alpha_params, args.lra, momentum=args.momentum,
                               weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_transform, valid_transform = data_transforms_cifar10()
    train_dataset = datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
      
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    print('========= initial architecture =========')
    if hasattr(model, 'module'):
        best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = model.module.fetch_best_arch()
    else:
        best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = model.fetch_best_arch()
    print('best model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M'.format(
        bitops, bita, bitw))
    print('expected model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M'.format(
        mixbitops, mixbita, mixbitw))

    for key, value in best_arch.items():
        print('{}: {}'.format(key, value))

    best_epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, arch_optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, arch_optimizer, epoch, args)

        print('========= architecture =========')
        if hasattr(model, 'module'):
            best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = model.module.fetch_best_arch()
        else:
            best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = model.fetch_best_arch()
        print('best model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M'.format(
            bitops, bita, bitw))
        print('expected model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M'.format(
            mixbitops, mixbita, mixbitw))
        for key, value in best_arch.items():
            print('{}: {}'.format(key, value))

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_epoch = epoch

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'arch_optimizer': arch_optimizer.state_dict(),
            }, is_best, epoch, args.step_epoch)

    print('Best Acc@1 {0} @ epoch {1}'.format(best_acc1, best_epoch))


def train(train_loader, model, criterion, optimizer, arch_optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    curr_lr = optimizer.param_groups[0]['lr']
    curr_lra = arch_optimizer.param_groups[0]['lr']
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}/{}]\t"
               "LR: {}\t"
               "LRA: {}\t".format(epoch, args.epochs, curr_lr, curr_lra))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        if args.arch=='DARTS':
            output, logit_aux = model(images)
        else:
            output = model(images)

        loss = criterion(output, target)

        if args.auxiliary:
          loss_aux = criterion(logit_aux, target)
          loss += args.auxiliary_weight*loss_aux

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # complexity penalty
        if args.complexity_decay != 0:
            if hasattr(model, 'module'):
                loss_complexity = args.complexity_decay * model.module.complexity_loss()
            else:
                loss_complexity = args.complexity_decay * model.complexity_loss()
            loss += loss_complexity

        # compute gradient and do SGD step
        optimizer.zero_grad()
        arch_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        arch_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.arch=='DARTS':
                output, logit_aux = model(images)
            else:
                output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, epoch, step_epoch, filename='arch_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'arch_model_best.pth.tar')
    if (epoch + 1) % step_epoch == 0:
        shutil.copyfile(filename, 'arch_checkpoint_ep{}.pth.tar'.format(epoch + 1))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, arch_optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every step_epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step_epoch))
    lra = args.lra * (0.1 ** (epoch // args.step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in arch_optimizer.param_groups:
        param_group['lr'] = lra


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
