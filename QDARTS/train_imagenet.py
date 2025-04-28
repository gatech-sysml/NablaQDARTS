import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
from pathlib import Path
from typing import List
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.autograd import Variable
from model import NetworkImageNet as Network
from model import QuantActivConv2d

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

parser = argparse.ArgumentParser("training imagenet")
parser.add_argument('--workers', type=int, default=32, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')
parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')
parser.add_argument('--trainfile', type=str, default='/serenity/data/datasets/Imagenet-FFCV/train_500_0.50_90.ffcv', help='train file path')
parser.add_argument('--valfile', type=str, default='/serenity/data/datasets/Imagenet-FFCV/val_500_0.50_90.ffcv', help='val file path')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dist', action='store_true', default=False)
parser.add_argument('--weights_path', type=str, default='../weights.pt', help='location of the model checkpoint')
parser.add_argument('--num_nodes', type=int, default=1)
parser.add_argument('--num_gpus_per_node', type=int, default=1)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--n_node', type=int, default=0)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--resume_path', type=str, default='./checkpoints/', help='path to checkpoint to resume from')
parser.add_argument('--allow8bit', action='store_true', default=False, help='Allow 8 bit in precision search space')

args, unparsed = parser.parse_known_args()

args.save = '{}eval-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

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

CLASSES = 1000

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def main_search(cfg):
    print('In main_search')
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)
    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ['LOCAL_RANK'])
    print('local_rank', local_rank)
    if cfg.dist:
        device = torch.device('cuda:%d' % local_rank) if cfg.dist else torch.device('cuda')
        torch.cuda.set_device(local_rank)
        world_size = args.num_nodes * args.num_gpus_per_node
        world_rank = args.n_node * args.num_gpus_per_node + local_rank
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=world_rank)
    else:
        torch.cuda.set_device(args.gpu)

    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------') 
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype, QuantActivConv2d, args.weights_path, device, allow8bitprec=args.allow8bit)
    
    if cfg.dist:
        model = model.cuda()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[local_rank, ],
                                                    output_device=local_rank, find_unused_parameters=True)
    elif num_gpus > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    if cfg.resume:
        checkpoint = torch.load(cfg.resume_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    logging.info('loaded checkpoints')

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.module.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    train_queue = create_train_loader(args.trainfile, num_workers=args.workers, batch_size=args.batch_size//args.num_gpus_per_node,
                        distributed=args.dist, in_memory=True, this_device=device)

    valid_queue = create_val_loader(args.valfile, num_workers=args.workers, batch_size=args.batch_size//args.num_gpus_per_node,
                        distributed=args.dist, this_device=device)

    if cfg.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_acc_top1 = 0
    best_acc_top5 = 0
    lr = args.learning_rate
    start_epoch = 0
    if cfg.resume:
        start_epoch = checkpoint['epoch'] + 1
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    for epoch in range(start_epoch, args.epochs):
        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            print('Wrong lr type, exit')
            sys.exit(1)
        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)

        if cfg.dist:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        epoch_start = time.time()

        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
        logging.info('Train_acc: %f', train_acc)

        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
        logging.info('Valid_acc_top1: %f', valid_acc_top1)
        logging.info('Valid_acc_top5: %f', valid_acc_top5)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds.', epoch_duration)
        is_best = False
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        if local_rank == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc_top1': best_acc_top1,
                'optimizer' : optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
                }, is_best, args.save)
        if epoch < 2:
            bitops, bita, bitw = model.module.fetch_arch_info()
            logging.info('Model complexity Bitops {}, bita {}, bitw {}'.format(bitops, bita, bitw))    
        
def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs -  epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr        

def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        b_start = time.time()
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step < 2:
            bitops, bita, bitw = model.module.fetch_arch_info()
            logging.info('Model complexity Bitops', str(bitops), 'bita', str(bita), 'bitw',str(bitw))

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs', 
                                    step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)

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
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg, duration)

    return top1.avg, top5.avg, objs.avg

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
