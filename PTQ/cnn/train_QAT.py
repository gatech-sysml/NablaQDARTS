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
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat as WeightQuant
from brevitas.quant import ShiftedUint8ActPerTensorFloat as ActQuant
import operations
import tqdm

    
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

##quantization parameters
parser.add_argument('--do_QAT', type=int, default=1, help="Perform post-training quantization or not")
parser.add_argument('--param_bits', type=int, default=4, help='bit-width for parameters')
parser.add_argument('--fwd_bits', type=int, default=8, help='bit-width for layer output')


args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

class ReducedRangeActQuant(ActQuant):
    bit_width = args.fwd_bits

class ReducedRangeWeightQuant(WeightQuant):
    bit_width = args.param_bits


CIFAR_CLASSES = 10

def turn_into_quant_aware_model(model):
    """
    model: torch model
    """
    act_quant    = ReducedRangeActQuant 
    weight_quant = ReducedRangeWeightQuant 
    
    for k, v in model._modules.items():
        layer =  model._modules[k]
        if len(layer._modules.keys()) != 0:
            turn_into_quant_aware_model(layer)
        elif isinstance(layer, torch.nn.Conv2d):
            
            layer = qnn.QuantConv2d(in_channels = layer.in_channels, out_channels = layer.out_channels,
            kernel_size = layer.kernel_size, stride = layer.stride, padding = layer.padding, 
            dilation = layer.dilation, groups = layer.groups, bias = layer.bias, padding_mode = layer.padding_mode, \
                weight_quant=weight_quant, output_quant=act_quant)
            print(f"Converting Conv2d with {args.param_bits} bit precision weight and {args.fwd_bits} bit precision activation")
            model._modules[k] = layer
        elif  isinstance(layer, torch.nn.BatchNorm2d):
            print("Not doing anything because its batch norm")
        elif isinstance(layer, torch.nn.ReLU):
            print("Not doing anything because its relu")
        elif isinstance(layer, operations.Identity):
            print("Not doing anything because its identity")
        elif isinstance(layer, torch.nn.modules.pooling.MaxPool2d):
            print("Not doing anything because its max pooling")
        elif isinstance(layer, torch.nn.modules.pooling.AvgPool2d):
            print("Not doing avg pool")
        elif isinstance(layer, torch.nn.modules.linear.Linear):
            # layer = qnn.QuantLinear(layer.in_features, layer.out_features, len(layer.bias) != 0, weight_quant=weight_quant,
            # output_quant=act_quant)
            # print(f"Converting Linear with {args.param_bits} bit precision weight and {args.fwd_bits} bit precision activation")
            # model._modules[k] = layer
            pass
        elif isinstance(layer, torch.nn.modules.pooling.AdaptiveAvgPool2d):
            print("Not doing anything because its adaptive avg pooling")
        else:
            print("Layer", layer, "not recognized")
    return model

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  if args.do_QAT:
      print("QAT enabled, converting model...")
      model = turn_into_quant_aware_model(model)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  for epoch in range(args.epochs):
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    
    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = input.cuda()
    target = target.cuda()

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  
  with torch.no_grad():
      for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()
    
        logits, _ = model(input)
        loss = criterion(logits, target)
    
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
    
        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

