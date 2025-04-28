import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import itertools

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedOp(nn.Module):

  def __init__(self, C, stride, conv_func, allow8bitprec):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2,2)
    OPS = get_ops(conv_func)
    for primitive in PRIMITIVES:
      op = OPS[primitive](C//2, stride, allow8bitprec=allow8bitprec, affine=False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C//2, affine=False))
      self._ops.append(op)

  def forward(self, x, weights, act, wt):
    
    dim_2 = x.shape[1]
    xtemp = x[ : , :  dim_2//2, :, :]
    xtemp2 = x[ : ,  dim_2//2:, :, :]
    xtemp3 = x[:,dim_2// 4:dim_2// 2, :, :]
    xtemp4 = x[:,dim_2// 2:, :, :]
    
    temp1 = sum(w.to(xtemp.device) * op(xtemp) if idx<3 else w * op(xtemp, act[idx-4], wt[idx-4]) 
                 for idx, (w, op) in enumerate(zip(weights, self._ops)))
    if temp1.shape[2] == x.shape[2]:
      ans = torch.cat([temp1,xtemp2],dim=1)
    else:
      ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)

    ans = channel_shuffle(ans,2)
    return ans

class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, conv_func, allow8bitprec):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, conv_func, affine=False, allow8bitprec=allow8bitprec)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, conv_func, affine=False, allow8bitprec=allow8bitprec)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, conv_func, affine=False, allow8bitprec=allow8bitprec)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride, conv_func, allow8bitprec)
        self._ops.append(op)

  def forward(self, s0, s1, weights,weights2, preact0, prewt0, preact1, prewt1, gamma_act, gamma_wt):
    s0 = self.preprocess0(s0, preact0, prewt0)
    s1 = self.preprocess1(s1, preact1, prewt1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(weights2[offset+j]*self._ops[offset+j](h, weights[offset+j], gamma_act[offset+j], gamma_wt[offset+j]) for j, h in enumerate(states))
      #s = channel_shuffle(s,4)
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, allow8bitprec=False):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.conv_func = MixActivConv2d
    self.quant_search_num = 3 if allow8bitprec else 2

    C_curr = stem_multiplier*C
    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr // 2, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )

 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = True
    self.num_red_cells = 0
    self.num_norm_cells = 0
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
        self.num_red_cells += 1
      else:
        reduction = False
        self.num_norm_cells += 1
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.conv_func, allow8bitprec)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()
    self._initialize_gammas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    red_ptr = 0
    norm_ptr = 0
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
        pre_act0 = F.softmax(self.gamma_activ_cellr_preprocess0[red_ptr], dim=-1)
        pre_wt0 = F.softmax(self.gamma_weight_cellr_preprocess0[red_ptr], dim=-1)
        pre_act1 = F.softmax(self.gamma_activ_cellr_preprocess1[red_ptr], dim=-1)
        pre_wt1 = F.softmax(self.gamma_weight_cellr_preprocess1[red_ptr], dim=-1)
        act = F.softmax(self.gamma_activ_cellr[red_ptr], dim=-1)
        wt = F.softmax(self.gamma_weight_cellr[red_ptr], dim=-1)   
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        red_ptr += 1
        for i in range(self._steps-1):
          end = start + n
          tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
        pre_act0 = F.softmax(self.gamma_activ_celln_preprocess0[norm_ptr], dim=-1)
        pre_wt0 = F.softmax(self.gamma_weight_celln_preprocess0[norm_ptr], dim=-1)
        pre_act1 = F.softmax(self.gamma_activ_celln_preprocess1[norm_ptr], dim=-1)
        pre_wt1 = F.softmax(self.gamma_weight_celln_preprocess1[norm_ptr], dim=-1)
        act = F.softmax(self.gamma_activ_celln[norm_ptr], dim=-1)
        wt = F.softmax(self.gamma_weight_celln[norm_ptr], dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
        norm_ptr += 1
        for i in range(self._steps-1):
          end = start + n
          tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      s0, s1 = s1, cell(s0, s1, weights,weights2, pre_act0, pre_wt0, pre_act1, pre_wt1, act, wt)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.betas_normal = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
    self.betas_reduce = Variable(1e-3*torch.randn(k).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
      self.betas_normal,
      self.betas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters
  
  def _initialize_gammas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.gamma_activ_celln_preprocess0 = Variable(1e-3*torch.randn(self.num_norm_cells, self.quant_search_num).cuda(), requires_grad=True)
    self.gamma_weight_celln_preprocess0 = Variable(1e-3*torch.randn(self.num_norm_cells, self.quant_search_num).cuda(), requires_grad=True)

    self.gamma_activ_celln_preprocess1 = Variable(1e-3*torch.randn(self.num_norm_cells, self.quant_search_num).cuda(), requires_grad=True)
    self.gamma_weight_celln_preprocess1 = Variable(1e-3*torch.randn(self.num_norm_cells, self.quant_search_num).cuda(), requires_grad=True)

    self.gamma_activ_celln = Variable(1e-3*torch.randn(self.num_norm_cells, k, num_ops-2, self.quant_search_num).cuda(), requires_grad=True) ## 5 conv operation among 8 search primitives
    self.gamma_weight_celln = Variable(1e-3*torch.randn(self.num_norm_cells, k, num_ops-2, self.quant_search_num).cuda(), requires_grad=True) ## 5 conv operation among 8 search primitives

    self.gamma_activ_cellr_preprocess0 = Variable(1e-3*torch.randn(self.num_red_cells, self.quant_search_num).cuda(), requires_grad=True)
    self.gamma_weight_cellr_preprocess0 = Variable(1e-3*torch.randn(self.num_red_cells, self.quant_search_num).cuda(), requires_grad=True)

    self.gamma_activ_cellr_preprocess1 = Variable(1e-3*torch.randn(self.num_red_cells, self.quant_search_num).cuda(), requires_grad=True)
    self.gamma_weight_cellr_preprocess1 = Variable(1e-3*torch.randn(self.num_red_cells, self.quant_search_num).cuda(), requires_grad=True)

    self.gamma_activ_cellr = Variable(1e-3*torch.randn(self.num_red_cells, k, num_ops-2, self.quant_search_num).cuda(), requires_grad=True) ## 5 conv operation among 8 search primitives
    self.gamma_weight_cellr = Variable(1e-3*torch.randn(self.num_red_cells, k, num_ops-2, self.quant_search_num).cuda(), requires_grad=True) ## 5 conv operation among 8 search primitives

    self._quant_parameters = [self.gamma_activ_celln_preprocess0, self.gamma_weight_celln_preprocess0, self.gamma_activ_celln_preprocess1, self.gamma_weight_celln_preprocess1, self.gamma_activ_celln, self.gamma_weight_celln,
                              self.gamma_activ_cellr_preprocess0, self.gamma_weight_cellr_preprocess0, self.gamma_activ_cellr_preprocess1, self.gamma_weight_cellr_preprocess1, self.gamma_activ_cellr, self.gamma_weight_cellr]

  def gamma_parameters(self):
    return self._quant_parameters
  
  def gamma_parameters_map(self):
    return {'gamma_activ_celln_preprocess0': self.gamma_activ_celln_preprocess0, 'gamma_weight_celln_preprocess0': self.gamma_weight_celln_preprocess0, 
            'gamma_activ_celln_preprocess1': self.gamma_activ_celln_preprocess1, 'gamma_weight_celln_preprocess1': self.gamma_weight_celln_preprocess1, 
            'gamma_activ_celln': self.gamma_activ_celln, 'gamma_weight_celln': self.gamma_weight_celln,
            'gamma_activ_cellr_preprocess0': self.gamma_activ_cellr_preprocess0, 'gamma_weight_cellr_preprocess0': self.gamma_weight_cellr_preprocess0, 
            'gamma_activ_cellr_preprocess1': self.gamma_activ_cellr_preprocess1, 'gamma_weight_cellr_preprocess1': self.gamma_weight_cellr_preprocess1, 
            'gamma_activ_cellr': self.gamma_activ_cellr, 'gamma_weight_cellr': self.gamma_weight_cellr}

  def get_gamma_params_iterable(self):
      gamma_params_map = self.gamma_parameters_map()
      return [gamma_params_map[k] for k in gamma_params_map.keys()]

  def get_gamma_params_namediterable(self):
      gamma_params_map = self.gamma_parameters_map()
      return [(k, gamma_params_map[k]) for k in gamma_params_map.keys()]

  def get_weight_precision_named_parameters(self):
      return itertools.chain(self.named_parameters(), self.get_gamma_params_namediterable())

  def get_weight_precision_parameters(self):
      return itertools.chain(self.parameters(), self.get_gamma_params_iterable())

  def genotype(self):

    def _parse(weights,weights2):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        W2 = weights2[start:end].copy()
        for j in range(n):
            W[j,:] = W[j,:]*W2[j]
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        
        #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene
    n = 3
    start = 2
    weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
    weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
    for i in range(self._steps-1):
      end = start + n
      #print(self.betas_reduce[start:end])
      tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
      tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
      start = end
      n += 1
      weightsr2 = torch.cat([weightsr2,tw2],dim=0)
      weightsn2 = torch.cat([weightsn2,tn2],dim=0)
    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),weightsn2.data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),weightsr2.data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def complexity_loss(self):
      size_product = []
      loss = 0
      for m in self.modules():
          if isinstance(m, self.conv_func):
              loss += m.complexity_loss()
              size_product += [m.size_product]
      normalizer = size_product[0].item()
      loss /= normalizer
      return loss

gaussian_steps = {1: 1.596, 2: 0.996, 3: 0.586, 4: 0.336}
hwgq_steps = {1: 0.799, 2: 0.538, 3: 0.3217, 4: 0.185}

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

class _gauss_quantize_resclaed_step(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        y = (torch.round(x/step+0.5)-0.5) * step
        thr = (lvls-0.5)*step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

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
        scale = range_vals / (max_8bit_int)
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

class MixActivConv2d(nn.Module):

  def __init__(self, inplane, outplane, wbits=[2,4], abits=[2,4], share_weight=True, **kwargs):
      super(MixActivConv2d, self).__init__()
      if kwargs['allow8bitprec']:
          self.wbits = [2, 4, 8]
          self.abits = [2, 4, 8]
          self.gamma_act = torch.tensor([0,0,0])
          self.gamma_wt = torch.tensor([0,0,0])
      else:
          self.wbits = wbits
          self.abits = abits
          self.gamma_act = torch.tensor([0,0])
          self.gamma_wt = torch.tensor([0,0])
      # build mix-precision branches
      self.mix_activ = MixQuantActiv(self.abits)
      self.share_weight = share_weight
      if share_weight:
          self.mix_weight = SharedMixQuantConv2d(inplane, outplane, self.wbits, **kwargs)
      else:
          raise ValueError('Cant find shared weight')
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



  def forward(self, input, gamma_act, gamma_wt):
      ## buffers the quant weights after each forward pass 
      self.gamma_act = gamma_act
      self.gamma_wt = gamma_wt
      in_shape = input.shape
      tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
      self.memory_size.copy_(tmp)
      tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
      self.size_product.copy_(tmp)
      out = self.mix_activ(input, gamma_act)
      out = self.mix_weight(out, gamma_wt)
      return out

  def complexity_loss(self):
      sw = self.gamma_act
      mix_abit = 0
      abits = self.mix_activ.bits
      for i in range(len(abits)):
          mix_abit += sw[i] * abits[i]
      sw = self.gamma_wt
      mix_wbit = 0
      wbits = self.mix_weight.bits
      for i in range(len(wbits)):
          mix_wbit += sw[i] * wbits[i]
      complexity = self.size_product.item() * mix_abit * mix_wbit
      return complexity

class MixQuantActiv(nn.Module):

    def __init__(self, bits):
        super(MixQuantActiv, self).__init__()
        self.bits = bits
        self.mix_activ = nn.ModuleList()
        for bit in self.bits:
            if bit < 8:
                self.mix_activ.append(HWGQ(bit=bit))
            elif bit < 32:
                self.mix_activ.append(QuantPerChannel(bit=bit))
            else:
                self.mix_activ.append(None)

    def forward(self, input, sw):
        outs = []
        for i, branch in enumerate(self.mix_activ):
            if branch == None:
                outs.append(input * sw[i])
            else:
                outs.append(branch(input) * sw[i])
        activ = sum(outs)
        return activ

class SharedMixQuantConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits, allow8bitprec, **kwargs):
        super(SharedMixQuantConv2d, self).__init__()
        assert not kwargs['bias']
        self.bits = bits
        self.conv = nn.Conv2d(inplane, outplane, **kwargs)
        self.steps = []
        for bit in self.bits:
            #assert 0 < bit < 32
            if bit < 8:
                self.steps.append(gaussian_steps[bit])
            else:
                self.steps.append(0)

    def forward(self, input, sw):
        mix_quant_weight = []
        conv = self.conv
        weight = conv.weight
        # save repeated std computation for shared weights
        weight_std = weight.std().item()
        for i, bit in enumerate(self.bits):
            if bit < 8:
                step = self.steps[i] * weight_std
                quant_weight = _gauss_quantize_resclaed_step.apply(weight, step, bit)
            elif bit < 32:
                quant_weight = _quant_per_channel.apply(weight, bit, False)            
            else:
                quant_weight = weight
            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        out = F.conv2d(
            input, mix_quant_weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return out
