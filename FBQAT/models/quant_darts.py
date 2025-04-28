import torch
from models.mixdarts import DARTS_CIFAR
from models import mixdarts
from models import quant_module as qm
from cnn import genotypes
from cnn import operations
#from DARTS_with_Post_Training_Quantization.cnn import genotypes
#from DARTS_with_Post_Training_Quantization.cnn import operations


def count_conv_layers(model):
    """
    counts the number of conv layers
    """
    num_conv_layers = 0
    for k, v in model._modules.items():
        layer =  model._modules[k]
        if layer == None:
            continue
        if len(layer._modules.keys()) != 0:
            num_conv_layers += count_conv_layers(layer)
        elif isinstance(layer, torch.nn.Conv2d):
            num_conv_layers += 1
    return num_conv_layers
 

def turn_into_quant_model(model, wbits, abits, precision_idx = [0]):
    """
    model: torch model
    bit: desired bit width
    """
    for k, v in model._modules.items():
        layer =  model._modules[k]
        if layer == None:
            continue
        if len(layer._modules.keys()) != 0:
            turn_into_quant_model(layer,wbits, abits, precision_idx)
        else:
            layer =  model._modules[k]
    
            if isinstance(layer, torch.nn.Conv2d):
                if layer.weight.grad == None:
                    del(layer)
                    model._modules[k] = None
                else:
                    layer = qm.QuantActivConv2d(inplane = layer.in_channels, outplane = layer.out_channels,
                    kernel_size = layer.kernel_size, stride = layer.stride[0], padding = layer.padding, 
                    dilation = layer.dilation, groups = layer.groups, bias = layer.bias, padding_mode = layer.padding_mode, 
                    wbit=wbits[precision_idx[0]], abit=abits[precision_idx[0]])
                    precision_idx[0]+=1
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
                print("Not doing anything because its linear")
            elif isinstance(layer, torch.nn.modules.pooling.AdaptiveAvgPool2d):
                print("Not doing anything because its adaptive avg pooling")
            else:
                print("Layer "+ str(layer)+ "not recognized")
                #raise(Exception("Layer"+ str(layer)+ "not recognized"))
    return model
    
import numpy as np
def remove_all_useless_layers(model):
    for k, v in model._modules.items():
        layer =  model._modules[k]
        if len(layer._modules.keys()) != 0:
            remove_all_useless_layers(layer)
        else:
            layer =  model._modules[k]
        
            if isinstance(layer, torch.nn.Conv2d):
                if layer.weight.grad == None:
                    del(layer)
                    model._modules[k] = None
                    print("Useless Layer")
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
                torch.nn.init.xavier_uniform(layer.weight)
            elif isinstance(layer, torch.nn.modules.pooling.AdaptiveAvgPool2d):
                print("Not doing anything because its adaptive avg pooling")
            else:
                print("Layer "+ str(layer)+ "not recognized")
                #raise(Exception("Layer"+ str(layer)+ "not recognized"))
    return model

def zero_all_weights(model):
    """
    model: torch model
    """
    for k, v in model._modules.items():
        layer =  model._modules[k]
        if layer == None:
            continue
            
        if len(layer._modules.keys()) != 0:
            zero_all_weights(layer)
        else:
            layer =  model._modules[k]
    
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform(layer.weight)
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
                torch.nn.init.xavier_uniform(layer.weight)
            elif isinstance(layer, torch.nn.modules.pooling.AdaptiveAvgPool2d):
                print("Not doing anything because its adaptive avg pooling")
            else:
                print("Layer "+ str(layer)+ "not recognized")
                #raise(Exception("Layer"+ str(layer)+ "not recognized"))
    return model


def _load_arch(arch_path, names_nbits):
    """
    This function is the same in all quant_<model>.py files
    """
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    best_arch, worst_arch = {}, {}
    for name in names_nbits.keys():
        best_arch[name], worst_arch[name] = [], []
    for name, params in state_dict.items():
        name = name.split('.')[-1]
        if name in names_nbits.keys():
            alpha = params.cpu().numpy()
            assert names_nbits[name] == alpha.shape[0] # <- this errors
            best_arch[name].append(alpha.argmax())
            worst_arch[name].append(alpha.argmin())
    return best_arch, worst_arch

def quantdarts_cfg(args, arch_cfg_path, **kwargs):
    wbits, abits = [2, 4], [2, 4]
    name_nbits = {'alpha_activ': len(abits), 'alpha_weight': len(wbits)}
    best_arch, worst_arch = _load_arch(arch_cfg_path, name_nbits)
    archas = [abits[a] for a in best_arch['alpha_activ']]
    archws = [wbits[w] for w in best_arch['alpha_weight']]
    model = DARTS_CIFAR(drop_path_prob = args.drop_path_prob, conv_func = qm.QuantActivConv2d, C=36, num_classes=10, layers = 20, auxiliary=True, genotype=genotypes.PCDARTS, device=device)

    # Do backward pass so we can identify which layers do not have a gradient and remove them
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    out, _ = model(torch.ones((5, 3, 32, 32)).cuda())
    ((abs(out)).sum() + (abs(_).sum())).backward()
    remove_all_useless_layers(model)  # remove all layers with unupdated gradient
    optimizer.zero_grad()


    zero_all_weights(model)
    new_model = turn_into_quant_model(model, archws, archas) # todo replace line with variant of load model
    return new_model


def fixed_quantdarts(args, device, abit = 4, wbit = 4):
    """
    Uses fixed bit quantization
    args: 
    """

    model = DARTS_CIFAR(drop_path_prob = args.drop_path_prob, device=device, conv_func = qm.QuantActivConv2d, C=48, num_classes=1000, layers = 14, auxiliary=True, genotype=genotypes.PCDARTS)

    # Do backward pass so we can identify which layers do not have a gradient and remove them
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    model.train()

    out, out_aux = model(torch.ones((5, 3, 224, 224)).cuda())
    ((abs(out)).sum() + abs(out_aux).sum() ).backward()
    remove_all_useless_layers(model)
    #optimizer.zero_grad()

    conv_layer_count = count_conv_layers(model)
    archas = [abit for i in range(conv_layer_count)]
    archws = [wbit for i in range(conv_layer_count)]


    print("Setting all weights to zero")
    new_model = turn_into_quant_model(model, archws, archas) # todo replace line with variant of load model
    optimizer.zero_grad()

    return new_model

def turn_into_mixed_quant_aware_model(model, wbits, abits):
    """
    model: torch model
    bit: desired bit width
    """
    for k, v in model._modules.items():
        layer =  model._modules[k]
        if layer == None:
            continue

        if len(layer._modules.keys()) != 0:
            turn_into_mixed_quant_aware_model(layer,wbits, abits)
        elif isinstance(layer, torch.nn.Conv2d):
            layer = qm.MixActivConv2d(inplane = layer.in_channels, outplane = layer.out_channels,
            kernel_size = layer.kernel_size, stride = layer.stride[0], padding = layer.padding, 
            dilation = layer.dilation, groups = layer.groups, bias = layer.bias, padding_mode = layer.padding_mode, 
            wbits=wbits, abits=abits, share_weight=True)
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
            pass
            # layer = qm.MixActivLinear(inplane = layer.in_features, outplane = layer.out_features,  bias = len(layer.bias) != 0,
            #                           wbits=wbits, abits=abits, share_weight=True)
            # # print("Converting Linear with bit precision", layer.bit)
            # model._modules[k] = layer
        elif isinstance(layer, torch.nn.modules.pooling.AdaptiveAvgPool2d):
            print("Not doing anything because its adaptive avg pooling")
        else:
            print("Layer", layer, "not recognized")
    return model