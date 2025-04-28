import torch
import torch.nn as nn

def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res


def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    in_c = input_size[1]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[:])

def calculate_norm(input_size):
    return torch.DoubleTensor([2 * input_size])


def calculate_linear(in_feature, num_elements):
    return torch.DoubleTensor([int(in_feature * num_elements)])

