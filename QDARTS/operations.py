import torch
import torch.nn as nn

def get_ops(conv_func):
  OPS = {
    'none' : lambda C, stride, allow8bitprec, affine: Zero(stride),
    'avg_pool_3x3' : lambda C, stride, allow8bitprec, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3' : lambda C, stride, allow8bitprec, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect' : lambda C, stride, allow8bitprec, affine: Identity() if stride == 1 else FactorizedReduce(C, C, conv_func, affine=affine, allow8bitprec=allow8bitprec),
    'sep_conv_3x3' : lambda C, stride, allow8bitprec, affine: SepConv(C, C, 3, stride, 1, conv_func, affine=affine, allow8bitprec=allow8bitprec),
    'sep_conv_5x5' : lambda C, stride, allow8bitprec, affine: SepConv(C, C, 5, stride, 2, conv_func, affine=affine, allow8bitprec=allow8bitprec),
    'sep_conv_7x7' : lambda C, stride, allow8bitprec, affine: SepConv(C, C, 7, stride, 3, conv_func, affine=affine, allow8bitprec=allow8bitprec),
    'dil_conv_3x3' : lambda C, stride, allow8bitprec, affine: DilConv(C, C, 3, stride, 2, 2, conv_func, affine=affine, allow8bitprec=allow8bitprec),
    'dil_conv_5x5' : lambda C, stride, allow8bitprec, affine: DilConv(C, C, 5, stride, 4, 2, conv_func, affine=affine, allow8bitprec=allow8bitprec),
    'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
      nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
      nn.BatchNorm2d(C, affine=affine)
      ),
  }
  return OPS

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, conv_func, affine=True, allow8bitprec=False):
    super(ReLUConvBN, self).__init__()
    self.relu = nn.ReLU(inplace=False)
    self.conv = conv_func(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, allow8bitprec=allow8bitprec)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x, act, wt):
    x = self.relu(x)
    x = self.conv(x, act, wt)
    x = self.bn(x)
    return x

class DilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation,conv_func, affine=True, allow8bitprec=False):
    super(DilConv, self).__init__()
    self.relu = nn.ReLU(inplace=False)
    self.conv1 = conv_func(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False, allow8bitprec=allow8bitprec)
    self.conv2 = conv_func(C_in, C_out, kernel_size=1, padding=0, bias=False, allow8bitprec=allow8bitprec)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x, act, wt):
    x = self.relu(x)
    x = self.conv1(x, act, wt)
    x = self.conv2(x, act, wt)
    x = self.bn(x)
    return x


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, conv_func, affine=True, allow8bitprec=False):
    super(SepConv, self).__init__()
    self.relu1 = nn.ReLU(inplace=False)
    self.conv1 = conv_func(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False, allow8bitprec=allow8bitprec)
    self.conv2 = conv_func(C_in, C_in, kernel_size=1, padding=0, bias=False, allow8bitprec=allow8bitprec)
    self.bn1 = nn.BatchNorm2d(C_in, affine=affine)
    self.relu2 = nn.ReLU(inplace=False)
    self.conv3 = conv_func(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False, allow8bitprec=allow8bitprec)
    self.conv4 = conv_func(C_in, C_out, kernel_size=1, padding=0, bias=False, allow8bitprec=allow8bitprec)
    self.bn2 = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x, act, wt):
    x = self.relu1(x)
    x = self.conv1(x, act, wt)
    x = self.conv2(x, act, wt)
    x = self.bn1(x)
    x = self.relu2(x)
    x = self.conv3(x, act, wt)
    x = self.conv4(x, act, wt)    
    x = self.bn2(x)
    return x


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x, act=None, w=None):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, conv_func, affine=True, allow8bitprec=False):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = conv_func(C_in, C_out // 2, kernel_size=1, stride=2, padding=0, bias=False, allow8bitprec=allow8bitprec)
    self.conv_2 = conv_func(C_in, C_out // 2, kernel_size=1, stride=2, padding=0, bias=False, allow8bitprec=allow8bitprec)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x, act, wt):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x, act, wt), self.conv_2(x[:,:,1:,1:], act, wt)], dim=1)
    out = self.bn(out)
    return out

