import torch.nn as nn
import math
from . import quant_module as qm
from cnn import genotypes
from cnn import operations
from cnn.model import NetworkImageNet
from cnn import utils

precision_idx = 0 #very weird way to keep track the number of conv layer

class DARTS_CIFAR(NetworkImageNet):

    def __init__(self, conv_func, drop_path_prob, device, **kwargs):

        super(DARTS_CIFAR, self).__init__(**kwargs)
        self.conv_func = conv_func
        self.drop_path_prob = drop_path_prob
        utils.load(self, "/serenity/scratch/Qdarts/QDARTS_per_cell_precisionweights/PC-DARTS/checkpoints/eval-try-20230401-032956/model_best.pth.tar", device)

    def complexity_loss(self):
        size_product = []
        loss =  complexity_loss(self, self.conv_func, size_product)
        normalizer = size_product[0].item()
        loss /= normalizer
        return loss
        
    def fetch_best_arch(self):
        best_arch, sum_bitops, sum_bita, sum_bitw, sum_mixbitops, sum_mixbita, sum_mixbitw, layer_idx =  fetch_best_arch(self, self.conv_func)
        return best_arch, sum_bitops, sum_bita, sum_bitw, sum_mixbitops, sum_mixbita, sum_mixbitw

    def fetch_arch_info(self):
        sum_bitops, sum_bita, sum_bitw, layer_idx = fetch_arch_info(self, self.conv_func)
        return sum_bitops, sum_bita, sum_bitw


def fetch_arch_info(model, conv_func, layer_idx = 0):
    sum_bitops = 0
    sum_bita = 0
    sum_bitw = 0
    
    for k, v in model._modules.items():
        layer =  model._modules[k]
        if layer == None:
            continue
        if isinstance(layer, conv_func):
            size_product = layer.size_product.item()
            memory_size = layer.memory_size.item()
            bitops = size_product * layer.abit * layer.wbit
            bita = layer.memory_size.item() * layer.abit
            bitw = layer.param_size * layer.wbit
            weight_shape = list(layer.conv.weight.shape)
            print('idx {} with shape {}, bitops: {:.3f}M * {} * {}, memory: {:.3f}K * {}, '
                    'param: {:.3f}M * {}'.format(layer_idx, weight_shape, size_product, layer.abit,
                                                layer.wbit, memory_size, layer.abit, layer.param_size, layer.wbit))

            sum_bitops += bitops
            sum_bita += bita
            sum_bitw += bitw
            layer_idx += 1

        elif len(layer._modules.keys()) != 0:
            bitops, bita, bitw, layer_idx = fetch_arch_info(layer, conv_func, layer_idx)
            sum_bitops += bitops
            sum_bita += bita
            sum_bitw += bitw
    return sum_bitops, sum_bita, sum_bitw, layer_idx


def complexity_loss(self, conv_func, size_product = []):
    """Unnormalized complexity_loss"""
    loss = 0
    for k, v in self._modules.items():
        layer =  self._modules[k]  
        if layer == None:
            continue  
        if isinstance(layer, conv_func):
            loss += layer.complexity_loss()
            size_product += [layer.size_product]
        elif len(layer._modules.keys()) != 0:
            loss += complexity_loss(layer, conv_func, size_product)
    return loss


def fetch_best_arch(self, conv_func, layer_idx = 0, best_arch = {}):
    sum_bitops = 0
    sum_bita = 0
    sum_bitw = 0
    sum_mixbitops = 0
    sum_mixbita = 0
    sum_mixbitw = 0
    
    """This is not recursive and therefore it should be wrong"""
    for k, v in self._modules.items():
        m =  self._modules[k]    
        if m == None:
            continue
        if isinstance(m, conv_func):
            layer_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = m.fetch_best_arch(layer_idx)
            for key in layer_arch.keys():
                if key not in best_arch:
                    best_arch[key] = layer_arch[key]
                else:
                    best_arch[key].append(layer_arch[key][0])

            sum_bitops += bitops
            sum_bita += bita
            sum_bitw += bitw
            sum_mixbitops += mixbitops
            sum_mixbita += mixbita
            sum_mixbitw += mixbitw
            layer_idx += 1

        elif len(m._modules.keys()) != 0:
            layer_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw, layer_idx = fetch_best_arch(m, conv_func, layer_idx, best_arch)
            for key in layer_arch.keys():
                if key not in best_arch:
                    best_arch[key] = layer_arch[key]
                else:
                    best_arch[key].append(layer_arch[key][0])

            sum_bitops += bitops
            sum_bita += bita
            sum_bitw += bitw
            sum_mixbitops += mixbitops
            sum_mixbita += mixbita
            sum_mixbitw += mixbitw            
            print("skipping None layer in fetch_best_arch" )

    return best_arch, sum_bitops, sum_bita, sum_bitw, sum_mixbitops, sum_mixbita, sum_mixbitw, layer_idx

