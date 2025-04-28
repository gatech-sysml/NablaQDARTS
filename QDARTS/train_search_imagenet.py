import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import copy
from pathlib import Path
from typing import List
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.autograd import Variable
from model_search_imagenet import Network
from architect import Architect

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder


parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--data', type=str, default='/tmp/cache/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--begin', type=int, default=35, help='batch size')

parser.add_argument('--tmp_data_dir', type=str, default='/cache/', help='temp data dir')
parser.add_argument('--trainfile', type=str, default='/serenity/data/datasets/Imagenet-FFCV/train_500_0.50_90.ffcv', help='train file path')
parser.add_argument('--valfile', type=str, default='/serenity/data/datasets/Imagenet-FFCV/val_500_0.50_90.ffcv', help='val file path')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dist', action='store_true', default=False)
parser.add_argument('--complexity-decay', '--cd', default=1e-4, type=float,
                    metavar='W', help='complexity decay (default: 1e-4)')
parser.add_argument('--q_lr', type=float, default=0.01)
parser.add_argument('--q_wd', type=float, default=1e-3)
parser.add_argument('--q_start', type=int, default=-1)
parser.add_argument('--num_nodes', type=int, default=1)
parser.add_argument('--num_gpus_per_node', type=int, default=1)
parser.add_argument('--local-rank', type=int, default=0)
parser.add_argument('--n_node', type=int, default=0)
parser.add_argument('--allow8bit', action='store_true', default=False, help='Allow 8 bit in precision search space')

args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '48495'

data_dir = os.path.join(args.tmp_data_dir, 'imagenet_search')
 #data preparation, we random sample 10% and 2.5% from training set(each class) as train and val, respectively.
#Note that the data sampling can not use torch.utils.data.sampler.SubsetRandomSampler as imagenet is too large
CLASSES = 1000


def main_search(cfg):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    
    logging.info("args = %s", args)
    
    local_rank = int(os.environ['LOCAL_RANK'])
    if cfg.dist:
        device = torch.device('cuda:%d' % local_rank) if cfg.dist else torch.device('cuda')
        torch.cuda.set_device(local_rank)
        world_size = args.num_nodes * args.num_gpus_per_node
        world_rank = args.n_node * args.num_gpus_per_node + local_rank
        dist.init_process_group(backend='nccl', init_method='env://',
            world_size=world_size, rank=world_rank)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    #dataset split

    model = Network(args.init_channels, CLASSES, args.layers, criterion, allow8bitprec=args.allow8bit)

    if cfg.dist:
        model = model.cuda()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank ], output_device=local_rank, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.module.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
               lr=args.arch_learning_rate, betas=(0.5, 0.999),
               weight_decay=args.arch_weight_decay)
    optimizer_q = torch.optim.SGD(model.module.gamma_parameters(), lr=args.q_lr, momentum=args.momentum, weight_decay=args.q_wd)

    train_queue = create_train_loader(args.trainfile, num_workers=args.workers, batch_size=args.batch_size//args.num_gpus_per_node,
                        distributed=args.dist, in_memory=True, this_device=device) 
    valid_queue = create_val_loader(args.valfile, num_workers=args.workers, batch_size=args.batch_size//args.num_gpus_per_node,
                        distributed=args.dist, this_device=device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, float(args.epochs), eta_min=args.learning_rate_min)

    #architect = Architect(model, args)
    lr=args.learning_rate
    for epoch in range(args.epochs):
        scheduler.step()
        scheduler_q.step()
        current_lr = scheduler.get_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
            print(optimizer)
        genotype = model.module.genotype()
        logging.info('genotype = %s', genotype)
        arch_param = model.module.arch_parameters()
        train_acc, train_obj = train(train_queue, valid_queue, model, optimizer, optimizer_a, optimizer_q, criterion, lr,epoch, cfg.complexity_decay)
        logging.info('Train_acc %f', train_acc)

        utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'optimizer_a' : optimizer_a.state_dict(),
                'optimizer_q' : optimizer_q.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scheduler_q': scheduler_q.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'gamma_params': model.module.gamma_parameters_map(),
                'arch_params': model.module.arch_parameters()
                }, False, args.save)
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        torch.save(model.module.gamma_parameters_map(), os.path.join(args.save, 'gamma_checkpoint.t7'))
        torch.save(model.module.arch_parameters(), os.path.join(args.save, 'arch_checkpoint.t7'))

        # validation
        if epoch> 47:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('Valid_acc %f', valid_acc)


def train(train_queue, valid_queue, model, optimizer, optimizer_a,optimizer_q, criterion, lr,epoch, complexity_decay):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)

        if epoch >=args.begin:
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.sum().backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()

        optimizer.zero_grad()
        optimizer_q.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss_complexity = 0
        if complexity_decay != 0:
            loss_complexity = complexity_decay * model.module.complexity_loss()
        loss += loss_complexity
        
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        optimizer.step()
        optimizer_q.step()


        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg



def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def create_train_loader(train_dataset_path, num_workers, batch_size,
                            distributed, in_memory, this_device):

    train_path = Path(train_dataset_path)
    assert train_path.is_file()

    res = 224 #self.get_resolution(epoch=0)
    decoder = RandomResizedCropRGBImageDecoder((res, res))
    image_pipeline: List[Operation] = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(this_device), non_blocking=True)
    ]

    order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
    loader = Loader(train_dataset_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=True,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed)

    return loader

def create_val_loader(val_dataset_path, num_workers, batch_size,
                        distributed, this_device, resolution=224):
    val_path = Path(val_dataset_path)
    assert val_path.is_file()
    res_tuple = (resolution, resolution)
    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(this_device),
        non_blocking=True)
    ]

    loader = Loader(val_dataset_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.SEQUENTIAL,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed)
    return loader

if __name__ == '__main__':
    main_search(cfg=args)
    #main()
