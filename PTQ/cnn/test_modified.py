import os
import sys
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


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')

##quantization parameters
parser.add_argument('--do_quant', type=int, default=0, help="Perform post-training quantization or not")
parser.add_argument('--quant_method', default='linear', help='linear|minmax|log|tanh')
parser.add_argument('--param_bits', type=int, default=4, help='bit-width for parameters')
parser.add_argument('--bn_bits', type=int, default=12, help='bit-width for running mean and std')
parser.add_argument('--fwd_bits', type=int, default=4, help='bit-width for layer output')
parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')
parser.add_argument('--n_sample', type=int, default=80, help='number of samples to infer the scaling factor')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10

import torch
from collections import OrderedDict
import math

def compute_integral_part(input, overflow_rate):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    if isinstance(v, Variable):
        v = float(v.data.cpu())
    sf = math.ceil(math.log2(v+1e-12))
    return sf

def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value

def log_minmax_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = min_max_quantize(input0, bits-1)
    v = torch.exp(v) * s
    return v

def log_linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input), 0.0, 0.0

    s = torch.sign(input)
    input0 = torch.log(torch.abs(input) + 1e-20)
    v = linear_quantize(input0, sf, bits-1)
    v = torch.exp(v) * s
    return v

def min_max_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    min_val, max_val = input.min(), input.max()

    if isinstance(min_val, Variable):
        max_val = float(max_val.data.cpu().numpy())
        min_val = float(min_val.data.cpu().numpy())

    input_rescale = (input - min_val) / (max_val - min_val)

    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n

    v =  v * (max_val - min_val) + min_val
    return v

def tanh_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    input = torch.tanh(input) # [-1, 1]
    input_rescale = (input + 1.0) / 2 #[0, 1]
    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n
    v = 2 * v - 1 # [-1, 1]

    v = 0.5 * torch.log((1 + v) / (1 - v)) # arctanh
    return v


class LinearQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LinearQuant, self).__init__()
        self.name = name
        self._counter = counter
        # print('asfasldkfnlksadnflkasnflksandfj')
        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate


    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            # print(self._counter)
            self._counter -= 1
            # print(self._counter)
            sf_new = self.bits - 1 - compute_integral_part(input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            output = linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)

class LogQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LogQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            log_abs_input = torch.log(torch.abs(input))
            sf_new = self.bits - 1 - compute_integral_part(log_abs_input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            output = log_linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)

class NormalQuant(nn.Module):
    def __init__(self, name, bits, quant_func):
        super(NormalQuant, self).__init__()
        self.name = name
        self.bits = bits
        self.quant_func = quant_func

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        output = self.quant_func(input, self.bits)
        return output

    def __repr__(self):
        return '{}(bits={})'.format(self.__class__.__name__, self.bits)

def duplicate_model_with_quant(model, bits, overflow_rate=0.0, counter=10, type='linear'):
    """assume that original model has at least a nn.Sequential"""
    assert type in ['linear', 'minmax', 'log', 'tanh']
    if isinstance(model, nn.Sequential):
        l = OrderedDict()
        for k, v in model._modules.items():
            # print(k)
            if isinstance(v, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.AvgPool2d)):
                l[k] = v
                if type == 'linear':
                    quant_layer = LinearQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
                elif type == 'log':
                    # quant_layer = LogQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=log_minmax_quantize)
                elif type == 'minmax':
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=min_max_quantize)
                else:
                    quant_layer = NormalQuant('{}_quant'.format(k), bits=bits, quant_func=tanh_quantize)
                l['{}_{}_quant'.format(k, type)] = quant_layer
            else:
                l[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter, type)
        m = nn.Sequential(l)
        return m
    else:
        for k, v in model._modules.items():
            # print(k)
            model._modules[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter, type)
        return model


def duplicate_model_with_quant_modified(model, bits, overflow_rate=0.0, counter=10, type='linear'):
    """assume that original model has at least a nn.Sequential"""
    assert type in ['linear', 'minmax', 'log', 'tanh']
    for k, v in model._modules.items():
        # print(k)
        model._modules[k] = duplicate_model_with_quant(v, bits, overflow_rate, counter, type)
    return model


def debug(model):
    if isinstance(model, (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d, nn.Linear)):
        print(type(model))
        print('quant')
        return
    else:
        for k, v in model._modules.items():
            debug(v)


def quantize_weights(args, model):
    # quantize parameters
    if args.param_bits <= 32:
        state_dict = model.state_dict()
        state_dict_quant = OrderedDict()
        for k, v in state_dict.items():
            if 'running' in k:
                if args.bn_bits >= 32:
                    # print("Ignoring {}".format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits = args.bn_bits
            else:
                bits = args.param_bits

            if args.quant_method == 'linear':
                sf = bits - 1. - compute_integral_part(v, overflow_rate=args.overflow_rate)
                v_quant = linear_quantize(v, sf, bits=bits)
            elif args.quant_method == 'log':
                v_quant = log_minmax_quantize(v, bits=bits)
            elif args.quant_method == 'minmax':
                v_quant = min_max_quantize(v, bits=bits)
            else:
                v_quant = tanh_quantize(v, bits=bits)
            state_dict_quant[k] = v_quant
            # print(k, bits)

        # model.load_state_dict(state_dict_quant)
        return state_dict_quant


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
  model = model.cuda()
  utils.load(model, args.model_path)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  _, test_transform = utils._data_transforms_cifar10(args)
  test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  model.drop_path_prob = args.drop_path_prob
  if args.do_quant==True:
      test_acc, test_obj = infer_with_quant(test_queue, model, criterion, args)
  else:
      test_acc, test_obj = infer(test_queue, model, criterion)
  logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for step, (input, target) in enumerate(test_queue):
      input = input.cuda()
      target = target.cuda()

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data, n)
      top1.update(prec1.data, n)
      top5.update(prec5.data, n)

      if step % args.report_freq == 0:
        logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def infer_with_quant(test_queue, model, criterion, args):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  state_dict_quant = quantize_weights(args, model)
  model.load_state_dict(state_dict_quant)
  if args.fwd_bits <= 32:
    model = duplicate_model_with_quant(model, bits=args.fwd_bits,
                                                          overflow_rate=args.overflow_rate,
                                                          counter=args.n_sample, type=args.quant_method)
  model.eval()
  with torch.no_grad():
    for step, (input, target) in enumerate(test_queue):
      input = input.cuda()
      target = target.cuda()

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data, n)
      top1.update(prec1.data, n)
      top5.update(prec5.data, n)

      if step % args.report_freq == 0:
        logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

