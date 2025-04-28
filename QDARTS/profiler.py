import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from collections.abc import Iterable

from helper import *
from model import QuantActivConv2d


def count_QconvNd(m, x, y: torch.Tensor):
    x = x[0]

    ret = calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = list(m.conv.kernel_size),
        groups = m.conv.groups,
        bias = m.conv.bias
    )
    return ret 

def count_bitaw(m: _ConvNd, x, y: torch.Tensor):
    bitw = x[0].shape[1] * y.shape[1] * m.kernel_size[0] * m.kernel_size[1]
    bita = x[0].shape[1] * x[0].shape[2] * x[0].shape[3] 
    return bita, bitw

def count_convNd(m: _ConvNd, x, y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    return calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = list(m.weight.shape[2:]),
        groups = m.groups,
        bias = m.bias
    )


def count_normalization(m: nn.modules.batchnorm._BatchNorm, x, y):
    # TODO: add test cases
    # https://github.com/Lyken17/pytorch-OpCounter/issues/124
    # y = (x - mean) / sqrt(eps + var) * weight + bias
    x = x[0]
    # bn is by default fused in inference
    flops = calculate_norm(x.numel())
    if (getattr(m, 'affine', False) or getattr(m, 'elementwise_affine', False)):
        flops *= 2
    return flops


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    num_elements = y.numel()

    return calculate_linear(total_mul, num_elements)


def countop(model, input_shape = (3, 224, 224)):
    bitops = 0
    mock_input = torch.rand(1, *input_shape)
    for name, module in model.named_modules():
        if not list(module.named_children()):
            if isinstance(module, nn.Conv2d):  # Assuming the leaf module you're looking for is of type nn.Conv2d
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                parent_module = dict(model.named_modules()).get(parent_name, None)
                w_bit = 4
                a_bit = 4
                if parent_module and hasattr(parent_module, 'w_bit') and hasattr(parent_module, 'a_bit'):
                    w_bit=parent_module.w_bit
                    a_bit=parent_module.a_bit
                
                
                shape_temp = (module.out_channels, int(mock_input.shape[2]/module.stride[0]), int(mock_input.shape[2]/module.stride[0]))

                output = torch.rand(1, *shape_temp)
  
                bitops += count_convNd(module, mock_input, output)*w_bit*a_bit

            elif isinstance(module, nn.BatchNorm2d):
                output = mock_input
                bitops += count_normalization(module, mock_input, output)

            elif isinstance(module, nn.Linear):
                output = mock_input  # Flatten for linear layer
                bitops += count_linear(module, mock_input, output)
            elif (module.__class__.__name__ == "QuantConv2d"):
                shape_temp = (module.out_channels, int(mock_input.shape[2]/module.stride[0]), int(mock_input.shape[2]/module.stride[0]))
                output = torch.rand(1, *shape_temp)
                bitops += count_QconvNd(module, mock_input, output)*module.w_bit*module.a_bit

            else:
                output = mock_input
        
            
            mock_input = output
            
    
    return bitops

def dfs_print(model, children_name='', layer_level=0):
    for name, layer in model.named_children():
        current_children_name = f'{children_name}.{name}' if children_name else name
        print('  ' * layer_level + f'Layer: {current_children_name}, Module: {layer}')



def clever_format(nums, format="%.3f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums


def print_layers_leaf(model, indent=''):
    for name, module in model.named_modules():
        if not list(module.named_children()):
            print(f'{indent}{module.__class__.__name__}: {module}')

            if isinstance(module, nn.Conv2d):  
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''

                parent_module = dict(model.named_modules()).get(parent_name, None)

                if parent_module and hasattr(parent_module, 'w_bit') and hasattr(parent_module, 'a_bit'):
                    print(f'{indent}  w_bit={parent_module.w_bit}, a_bit={parent_module.a_bit}')



def io_size(model):

    def register_hook(module):
        def hook(module, input, output):
            print(f'{module.__class__.__name__}:')
            print(f'Input size: {list(input[0].size())}')
            print(f'Output size: {list(output.size())}')
            print('---') 
            module.register_forward_hook(hook)

    model.apply(register_hook)



def register_forward_hooks(model):
    leaf_layers = []
    bit_ops = 0
    bit_act = 0
    bit_wgt = 0
    def update_count(bita, bitw, bitops):
        nonlocal bit_act, bit_wgt, bit_ops  # Allows us to modify the outer variable
        print("Layer BitAs", clever_format([bita]))
        bit_act += bita
        print("Total BitAs", clever_format([bit_act]))
        print("Layer BitWs", clever_format([bitw]))
        bit_wgt += bitw
        print("Total BitWs", clever_format([bit_wgt]))

        print("Layer BitOps", clever_format([bitops]))
        bit_ops += bitops
        print("Total BitOps", clever_format([bit_ops]))
        print()
        print()



    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, QuantActivConv2d, nn.BatchNorm2d, nn.Conv2d)):
            if(name.startswith("stem")):
                print('layer', layer)
                continue
            def forward_hook(layer_name):
                def hook(module, input, output):
                    print(f"Layer: {layer_name}")
                    print(f"Input Size: {list(input[0].size())}")
                    print(f"Output Size: {list(output.size())}")
                    if isinstance(module, QuantActivConv2d):
                        print(module)
                        print(f"abit: {module.abit}")
                        print(f"wbit: {module.wbit}")
                        bita, bitw = count_bitaw(module.conv, input, output)
                        bitops = count_QconvNd(module, input, output)
                        bita *= module.abit
                        bitw *= module.wbit
                        bitops *= module.abit * module.wbit

                    else:
                        bita = 0
                        bitw = 0
                        bitops = 0
                    update_count(bita, bitw, bitops)

                return hook

            handle = layer.register_forward_hook(forward_hook(name))
            leaf_layers.append((name, layer, handle))
    return leaf_layers, bit_ops
