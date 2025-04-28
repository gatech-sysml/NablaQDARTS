import torch
import torch.nn as nn
from operations_eval import *
from torch.autograd import Variable
from utils import drop_path
import numpy as np


class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, conv_func, 
                  preact0, prewt0, preact1, prewt1, act, wt, wbits_list, abits_list):
    super(Cell, self).__init__()
    self.OPS = get_ops_eval(conv_func=conv_func)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, conv_func=conv_func, abit =preact0, wbit = prewt0)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, conv_func=conv_func, abit =preact0, wbit = prewt0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, conv_func=conv_func, abit =preact1, wbit = prewt1)
    
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat

    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for idx, (name, index) in enumerate(zip(op_names, indices)):
      stride = 2 if reduction and index < 2 else 1
      if idx<2:
          edge_idx = index
      elif idx<4:
          edge_idx = index + 2
      elif idx<6:
          edge_idx = index + 5
      else:
          edge_idx = index + 9
      
      if name in Conv_PRIMITIVES:
          wb = wbits_list[wt[edge_idx ,Conv_PRIMITIVES.index(name)]]
          ab = abits_list[act[edge_idx ,Conv_PRIMITIVES.index(name)]]
          op = self.OPS[name](C, stride, affine=True, a=ab, w=wb)
      else:
          op = self.OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes, conv_fn = nn.Conv2d):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      conv_fn(C, 128, kernel_size=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      conv_fn(128, 768, kernel_size=2, bias=False),
      # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
      # Commenting it out for consistency with the experiments in the paper.
      # nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, conv_func, weights_path, allow8bitprec=False):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self.conv_func = conv_func

    if allow8bitprec:
        wbits, abits = [2, 4, 8], [2, 4, 8]
        gamma_len = 3
    else:
        wbits, abits = [2, 4], [2, 4]
        gamma_len = 2
    state_dict = torch.load(weights_path)
    
    preact0_n = state_dict['gamma_activ_celln_preprocess0'].cpu().detach().numpy().argmax(axis=-1)
    prewt0_n = state_dict['gamma_weight_celln_preprocess0'].cpu().detach().numpy().argmax(axis=-1)
    preact1_n = state_dict['gamma_activ_celln_preprocess1'].cpu().detach().numpy().argmax(axis=-1)
    prewt1_n = state_dict['gamma_weight_celln_preprocess1'].cpu().detach().numpy().argmax(axis=-1)

    preact0_r = state_dict['gamma_activ_cellr_preprocess0'].cpu().detach().numpy().argmax(axis=-1)
    prewt0_r = state_dict['gamma_weight_cellr_preprocess0'].cpu().detach().numpy().argmax(axis=-1)
    preact1_r = state_dict['gamma_activ_cellr_preprocess1'].cpu().detach().numpy().argmax(axis=-1)
    prewt1_r = state_dict['gamma_weight_cellr_preprocess1'].cpu().detach().numpy().argmax(axis=-1) 

    gamma_weight_n = state_dict['gamma_weight_celln']   
    gamma_weight_r = state_dict['gamma_weight_cellr']   
    gamma_activ_n = state_dict['gamma_activ_celln']   
    gamma_activ_r = state_dict['gamma_activ_cellr']   
    
    w_bits_n = reorganize(gamma_weight_n, gamma_len)
    w_bits_r = reorganize(gamma_weight_r, gamma_len)
    a_bits_n = reorganize(gamma_activ_n, gamma_len)
    a_bits_r = reorganize(gamma_activ_r, gamma_len)

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    red_ptr = 0
    norm_ptr = 0
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        if i > 9:
            red_ptr = 0
        C_curr *= 2
        reduction = True
        cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.conv_func, 
                  preact0=abits[preact0_r[red_ptr]], prewt0=wbits[prewt0_r[red_ptr]], preact1=abits[preact1_r[red_ptr]], prewt1=wbits[prewt1_r[red_ptr]], 
                  act = a_bits_r[red_ptr], wt = w_bits_r[red_ptr], wbits_list=wbits, abits_list=abits)
        red_ptr += 1
      else:
        if i > 6:
            norm_ptr = 0
        reduction = False
        cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.conv_func, 
                  preact0=abits[preact0_n[norm_ptr]], prewt0=wbits[prewt0_n[norm_ptr]], preact1=abits[preact1_n[norm_ptr]], prewt1=wbits[prewt1_n[norm_ptr]], 
                  act = a_bits_n[norm_ptr], wt = w_bits_n[norm_ptr], wbits_list=wbits, abits_list=abits)
        norm_ptr += 1
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
      if i == 2*layers//3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2*self._layers//3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits, logits_aux
  
  def fetch_arch_info(self):
      sum_bitops, sum_bita, sum_bitw = 0, 0, 0
      layer_idx = 0
      for (name, m) in self.named_modules():
          if isinstance(m, self.conv_func):
              size_product = m.size_product.item()
              memory_size = m.memory_size.item()
              bitops = size_product * m.abit * m.wbit
              bita = m.memory_size.item() * m.abit
              bitw = m.param_size * m.wbit
              weight_shape = list(m.conv.weight.shape)
              print('name {} idx {} with shape {}, bitops: {:.3f}M * {} * {}, memory: {:.3f}K * {}, '
                    'param: {:.3f}M * {}'.format(name, layer_idx, weight_shape, size_product, m.abit,
                                                  m.wbit, memory_size, m.abit, m.param_size, m.wbit))
              sum_bitops += bitops
              sum_bita += bita
              sum_bitw += bitw
              layer_idx += 1
      return sum_bitops, sum_bita, sum_bitw

class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, auxiliary, genotype, conv_func, weights_path, device, allow8bitprec=False):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    self.conv_func = conv_func

    if allow8bitprec:
        wbits, abits = [2, 4, 8], [2, 4, 8]
        gamma_len = 3
    else:
        wbits, abits = [2, 4], [2, 4]
        gamma_len = 2
    state_dict = torch.load(weights_path, map_location=device)

    preact0_n = state_dict['gamma_activ_celln_preprocess0'].cpu().detach().numpy().argmax(axis=-1)
    prewt0_n = state_dict['gamma_weight_celln_preprocess0'].cpu().detach().numpy().argmax(axis=-1)
    preact1_n = state_dict['gamma_activ_celln_preprocess1'].cpu().detach().numpy().argmax(axis=-1)
    prewt1_n = state_dict['gamma_weight_celln_preprocess1'].cpu().detach().numpy().argmax(axis=-1)   

    preact0_r = state_dict['gamma_activ_cellr_preprocess0'].cpu().detach().numpy().argmax(axis=-1)
    prewt0_r = state_dict['gamma_weight_cellr_preprocess0'].cpu().detach().numpy().argmax(axis=-1)
    preact1_r = state_dict['gamma_activ_cellr_preprocess1'].cpu().detach().numpy().argmax(axis=-1)
    prewt1_r = state_dict['gamma_weight_cellr_preprocess1'].cpu().detach().numpy().argmax(axis=-1) 

    gamma_weight_n = state_dict['gamma_weight_celln']   
    gamma_weight_r = state_dict['gamma_weight_cellr']   
    gamma_activ_n = state_dict['gamma_activ_celln']   
    gamma_activ_r = state_dict['gamma_activ_cellr']   

    w_bits_n = reorganize(gamma_weight_n, gamma_len)
    w_bits_r = reorganize(gamma_weight_r, gamma_len)
    a_bits_n = reorganize(gamma_activ_n, gamma_len)
    a_bits_r = reorganize(gamma_activ_r, gamma_len)

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    red_ptr = 0
    norm_ptr = 0
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        if i > 9:
            red_ptr = 0
        C_curr *= 2
        reduction = True
        cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.conv_func, 
                  preact0=abits[preact0_r[red_ptr]], prewt0=wbits[prewt0_r[red_ptr]], preact1=abits[preact1_r[red_ptr]], prewt1=wbits[prewt1_r[red_ptr]], 
                  act = a_bits_r[red_ptr], wt = w_bits_r[red_ptr], wbits_list=wbits, abits_list=abits)
        red_ptr += 1
      else:
        if i > 6:
            norm_ptr = 0
        reduction = False
        cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.conv_func, 
                  preact0=abits[preact0_n[norm_ptr]], prewt0=wbits[prewt0_n[norm_ptr]], preact1=abits[preact1_n[norm_ptr]], prewt1=wbits[prewt1_n[norm_ptr]], 
                  act = a_bits_n[norm_ptr], wt = w_bits_n[norm_ptr], wbits_list=wbits, abits_list=abits)
        norm_ptr += 1
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes, conv_fn=Conv2dWrapper)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux
  
  def fetch_arch_info(self):
      sum_bitops, sum_bita, sum_bitw = 0, 0, 0
      layer_idx = 0
      sum_bitops_cells = [0]*self._layers
      sum_bita_cells = [0]*self._layers
      sum_bitw_cells = [0]*self._layers
      for (name, m) in self.named_modules():
          if isinstance(m, self.conv_func):
              size_product = m.size_product.item()
              memory_size = m.memory_size.item()
              bitops = size_product * m.abit * m.wbit
              bita = m.memory_size.item() * m.abit
              bitw = m.param_size * m.wbit
              weight_shape = list(m.conv.weight.shape)
              sum_bitops += bitops
              sum_bita += bita
              sum_bitw += bitw
              cell_id = int(name.split('.')[1])
              sum_bitops_cells[cell_id] += bitops
              sum_bita_cells[cell_id] += bita
              sum_bitw_cells[cell_id] += bitw
              layer_idx += 1
          if isinstance(m, Conv2dWrapper):
              size_product = m.size_product.item()
              memory_size = m.memory_size.item()
              bitops = size_product * 32 * 32 #m.abit * m.wbit
              bita = m.memory_size.item() * 32 #m.abit
              bitw = m.param_size * 32 #m.wbit
              weight_shape = list(m.conv.weight.shape)
              sum_bitops += bitops
              sum_bita += bita
              sum_bitw += bitw
              layer_idx += 1
      return sum_bitops, sum_bita, sum_bitw

def reorganize(arr, gamma_len):
    arr = arr.cpu().detach().numpy()
    assert arr.shape == (6,14,6,gamma_len) or arr.shape ==(2,14,6,gamma_len)
    arr_new = np.zeros((arr.shape[0],14,5,gamma_len))
    arr_new[:,:,0,:] = arr[:,:,-1,:]
    arr_new[:,:,1:5,:] = arr[:,:,0:4,:]
    arr_new = arr_new.argmax(-1)
    return arr_new

hwgq_steps = {1: 0.799, 2: 0.538, 3: 0.3217, 4: 0.185}
gaussian_steps = {1: 1.596, 2: 0.996, 3: 0.586, 4: 0.336}

class _gauss_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        gamma = x.std().item()
        step *= gamma
        y = (torch.round(x/step+0.5)-0.5) * step
        thr = (lvls-0.5)*step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class _hwgq(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step):
        y = torch.round(x / step) * step
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class HWGQ(nn.Module):
    def __init__(self, bit=2):
        super(HWGQ, self).__init__()
        self.bit = bit
        if bit < 32:
            self.step = hwgq_steps[bit]
        else:
            self.step = None

    def forward(self, x):
        if self.bit >= 32:
            return x.clamp(min=0.0)
        lvls = float(2 ** self.bit - 1)
        clip_thr = self.step * lvls
        y = x.clamp(min=0.0, max=clip_thr)
        return _hwgq.apply(y, self.step)
    
class QuantPerChannel(nn.Module):
    def __init__(self, bit=8):
        super(QuantPerChannel, self).__init__()
        self.bit = bit

    def forward(self, x):
        if self.bit >= 32:
            return x.clamp(min=0.0)
        return _quant_per_channel.apply(x, self.bit)
    
class _quant_per_channel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, num_bits=8, posonly=True, channel_axis=1):

        min_vals = torch.amin(x, dim=(0, 2, 3), keepdim=True)
        max_vals = torch.amax(x, dim=(0, 2, 3), keepdim=True)

        if posonly:
            posonly = (torch.min(x).item() >= -0.05)

        # compute the range of each channel
        if posonly:        
            range_vals = max_vals - min_vals
        else:
            range_vals = 2 * torch.max(torch.abs(max_vals), torch.abs(min_vals))

        # compute the scale and zero-point values for each channel
        max_8bit_int = 2 ** num_bits - 1
        scale = range_vals / (max_8bit_int) + 1e-10
        zero_point = torch.round(-min_vals / scale)
        zero_point = torch.clamp(zero_point, 0, max_8bit_int)

        # quantize the input tensor using per-channel scale and zero-point values
        if posonly:
            y = (torch.round(x / scale)) * scale
            thr = (max_8bit_int) * scale
            y = torch.clamp(y, min=torch.tensor(0.0).cuda(), max=thr)
        else:
            lvls = max_8bit_int/2
            y = (torch.round(x / scale + 0.5) - 0.5) * scale
            thr = (lvls-0.5) * scale
            y = torch.clamp(y, min=-thr, max=thr)

        quant_error = torch.abs(x - y)
        if torch.max(quant_error).item() > 3:
            print('8bit', 'min_error', torch.min(quant_error).item(), 'max_error', torch.max(quant_error).item(), 'avg_error', torch.mean(quant_error).item(), 'max_scale', torch.max(scale).item(), 'min_x', torch.min(x).item(), 'max_x', torch.max(x).item(), 'max_thr', torch.max(thr).item(), 'posonly', posonly )

        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

    
class QuantConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        self.bit = kwargs.pop('bit', 1)
        super(QuantConv2d, self).__init__(*kargs, **kwargs)
        assert self.bit > 0
        if self.bit < 8:
            self.step = gaussian_steps[self.bit]
        else:
            self.step = 0

    def forward(self, input):
        # quantized conv, otherwise regular
        if self.bit < 32:
            assert self.bias is None
            if self.bit < 8:
                quant_weight = _gauss_quantize.apply(self.weight, self.step, self.bit)
            else:
                quant_weight = _quant_per_channel.apply(self.weight, self.bit, False)
            out = nn.functional.conv2d(
                input, quant_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            out = nn.functional.conv2d(
                input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

class QuantActivConv2d(nn.Module):

    def __init__(self, inplane, outplane, wbit=1, abit=2, **kwargs):
        super(QuantActivConv2d, self).__init__()
        self.abit = abit
        self.wbit = wbit
        if self.abit < 8:
            self.activ = HWGQ(abit)
        elif self.abit < 32:
            self.activ = QuantPerChannel(abit)
        else:
            self.activ = None
        self.conv = QuantConv2d(inplane, outplane, bit=wbit, **kwargs)
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        self.size_product.copy_(tmp)
        if self.activ == None:
            out = input
        else:
            out = self.activ(input)
        out = self.conv(out)
        return out

class Conv2dWrapper(nn.Module):

    def __init__(self, inplane, outplane, **kwargs):
        super(Conv2dWrapper, self).__init__()
        self.conv = nn.Conv2d(inplane, outplane, **kwargs)
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        self.size_product.copy_(tmp)
        #out = self.activ(input)
        out = self.conv(input)
        return out
