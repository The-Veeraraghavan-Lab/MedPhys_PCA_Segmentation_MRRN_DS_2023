
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .incre_MRRN import Incre_MRRN_deepsup


def get_Incre_MRRN_deepsup (n_channels,n_class,init_type='normal',gpu_ids=[], deeplayer=0):
    net_Unet=None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    net_Unet=Incre_MRRN_deepsup(n_channels, n_class, deeplayer)
    #if use_gpu:
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    net_Unet.to(device)
    #net_Unet.cuda()
    init_weights(net_Unet, init_type=init_type)

    return net_Unet


def weights_init_normal(m):
    classname = m.__class__.__name__
    # if not(hasattr(m, 'weight')):
    #     for it in range(0,np.size(m)):
    #         # print(classname)
    #         if classname.find('Conv') != -1:
    #             #print (m)
    #             k=m[it]
    #             init.normal_(k.weight.data, 0.0, 0.02)
    #         elif classname.find('Linear') != -1:
    #             k=m[it]
    #             init.normal(k.weight.data, 0.0, 0.02)
    #         elif classname.find('BatchNorm2d') != -1:
    #             k=m[it]
    #             init.normal(k.weight.data, 1.0, 0.02)
    #             init.constant(k.bias.data, 0.0)
    # else:
    # print(classname)
    if classname.find('Conv') != -1:
        #print (m)
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)  #niter_decay_100 #niter 100
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
