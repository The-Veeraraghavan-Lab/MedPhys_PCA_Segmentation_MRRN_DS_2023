import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from math import floor,isnan
from options.seg_options import SegmentationOptions

import torch.nn.functional as F

import torch.nn as nn

opt = SegmentationOptions().parse()
num_organ = 1
imsize = opt.fineSize #448


class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        self.ones = torch.eye(depth).to(device)

    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        smooth = 0.0001
        #print(np.shape(input))
        batch_size = input.size(0)
        
        # valid_mask = target.ne(-1)
        # target = target.masked_fill_(~valid_mask, 0)
        
        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        #target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        #target = F.softmax(target, dim=1).view(batch_size, self.n_classes, -1)
        target=torch.cat((1-target,target),1)
        target=target.contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input * target , 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score


class DiceLoss_test(nn.Module):
    def __init__(self,num_organ=6):
        super(DiceLoss_test, self).__init__()

        self.num_organ=num_organ
    def forward(self, pred_stage1, target):
        """
        :param pred_stage1: (B, 9,  256, 256)
        :param pred_stage2: (B, 9, 256, 256)
        :param target: (B, 256, 256)
        :return: Dice
        """
        pred_stage1 = F.softmax(pred_stage1, dim=1)
        num_organ=1
        # 
        #[b,12,256,256]
        organ_target = torch.zeros((target.size(0), num_organ, imsize, imsize))
        #print (organ_target.size())
        #[0-11] 
        for organ_index in range(0, num_organ ):
            #print (organ_index)
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            #print (organ_target[:, organ_index, :, :].size())
            #print (temp_target.size())
            organ_target[:, organ_index, :, :] = torch.squeeze(temp_target)
            # organ_target: (B, 8,  128, 128)

        organ_target = organ_target.cuda()

        # loss
        dice_stage1 = 0.0

        for organ_index in range(0, num_organ ):
            dice_stage1 += 2 * (pred_stage1[:, organ_index, :, :] * organ_target[:, organ_index , :, :]).sum(dim=1).sum(
                dim=1) / (pred_stage1[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) +
                          organ_target[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) + 1e-5)

        
        dice_stage1 /= num_organ


        # 
        dice = dice_stage1 

        # 
        return (1 - dice).mean()

import numpy as np

class MRRN_Segmentor(BaseModel):
    def name(self):
        return 'MRRN_Segmentor'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size) # input A
        if(opt.model_type == 'classifier'):
           self.input_A_y = self.Tensor(nb, opt.output_nc, 1, 1)
        else:
           self.input_A_y = self.Tensor(nb, opt.output_nc, size, size) # input B
           
           
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
           
           
        self.input_A=self.input_A.to(device)
        self.input_A_y=self.input_A_y.to(device)
        if(opt.model_type == 'multi'):
          #self.input_A_z = self.Tensor(3,1)
          ## going to treat this as a pixel-wise output within the mask -- makes sense if it is regarding the pCR because the whole tumor part tumor may respond/ or for subtype
          self.input_A_z = self.Tensor(nb, opt.output_nc, size, size)
          self.input_A_z = self.input_A_z.to(device)

        self.test_A = self.Tensor(nb, opt.output_nc, size, size) # input B
        self.num_organ=1 #6#9+3 #/->6 ->8 

        self.hdicetest=DiceLoss_test()
        self.dicetest=SoftDiceLoss()
        #MRRN
        if(opt.model_type == 'deep'):
           self.netSeg_A=networks.get_Incre_MRRN_deepsup(opt.nchannels,1,opt.init_type,self.gpu_ids, opt.deeplayer)

        #flops, params = get_model_complexity_info(self.netSeg_A, (256, 256), as_strings=True, print_per_layer_stat=True)
        #print ('params is ',params)
        #print ('flops is ',flops)
        #self.criterion = nn.CrossEntropyLoss()

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            if self.isTrain:
                self.load_network(self.netSeg_A,'Seg_A',which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.optimizer_Seg_A = torch.optim.Adam(self.netSeg_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),amsgrad=True)
            if opt.optimizer == 'SGD':
                self.optimizer_Seg_A = torch.optim.SGD(self.netSeg_A.parameters(), lr=opt.lr, momentum=0.99)
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_Seg_A)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        if self.isTrain:
            networks.print_network(self.netSeg_A)
        print('-----------------------------------------------')

    def set_test_input(self,input):
        input_A1=input[0]
        self.test_A,self.test_A_y=torch.split(input_A1, input_A1.size(0), dim=1)    
        
    def cross_entropy_2D(self, input, target, weight=None, size_average=True):
        n, c, h, w = input.size()
        input = input.float()
        if 1 < 0:
           loss = nn.CrossEntropyLoss(F.log_softmax(input), target.long())
        else:
           log_p = F.log_softmax(input, dim=1)
           log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
           target = target.view(target.numel())
           target = target.long()
           loss = F.nll_loss(log_p, target, weight=weight, size_average=True)
        #if size_average:
        #    loss /= float(target.numel())
        return loss

    def cross_entropy_1D(self, input, target, weight=None, size_average=True):
        input = input.float()
        if 1 < 0:
           loss = nn.CrossEntropyLoss(F.log_softmax(input), target.long())
        else:
           log_p = F.log_softmax(input, dim=1)
           target = target.view(target.numel())
           target = target.long()
           loss = F.nll_loss(log_p, target, weight=weight, size_average=True)
        return loss


    def dice_loss(self,input, target,x):

        ##USE for CrossEntrophy
        CE_loss=0
        CE_ohm_loss=0

        
        Soft_dsc_loss=0
        dice_test=0
        hdice_test=0
        dice_ce=0
        CE_loss = 0   
        if(self.opt.loss == 'dice_ce'):
           dice_ce = 1
        if(self.opt.loss == 'ce'):
           CE_loss = 1
        if(self.opt.loss == 'soft_dsc'):
           Soft_dsc_loss = 1
        if(self.opt.loss == 'dice'):
           dice_test = 1
        if(self.opt.loss == 'hdice'):
           hdice_test = 1
        if(self.opt.model_type == 'classification'):
           CE_loss = 1
           dice_ce = 0

        if CE_loss:    
            #print(input.size())
            n, c, h, w = input.size()
            input=input.float()
            log_p = F.log_softmax(input,dim=1)
            log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
            target = target.view(target.numel())
            target=target.long()
            loss = F.nll_loss(log_p, target, weight=None, size_average=True)
            #size_average=False
            #if size_average:
            #    loss /= float(target.numel())
            
        elif Soft_dsc_loss:       
            loss=self.soft_dice_loss(input,target)                
        elif CE_ohm_loss:
            loss=self.CrossEntropy2d_Ohem(input,target)
        elif dice_test:
            loss=self.dicetest(input, target)    
        elif hdice_test:
            loss=self.hdicetest(input, target)    
        else: #dice_ce
           
            loss1=self.dicetest(input, target)   
            n, c, h, w = input.size()
            input=input.float()
            log_p = F.log_softmax(input,dim=1)
            log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
            target = target.view(target.numel())
            target=target.long()
            loss2 = F.nll_loss(log_p, target, weight=None, size_average=True)
        
            
            loss=0.5*loss1+0.5*loss2
            
        return loss

    def get_curr_lr(self):
        self.cur_lr=self.optimizer_Seg_A.param_groups[0]['lr'] 

        return self.cur_lr
    
    def cal_dice_loss(self,pred_stage1, target):
        """
        :param pred_stage1: (B, 9,  256, 256)
        :param pred_stage2: (B, 9, 256, 256)
        :param target: (B, 256, 256)
        :return: Dice
        : HV note: These parameters are a relic -- don't make any sense
        """
        num_organ=pred_stage1.size(1)-1
        
        organ_target = torch.zeros((target.size(0), num_organ+1, imsize, imsize))  # 8+1
        pred_stage1=F.softmax(pred_stage1,dim=1)
        

        for organ_index in range(num_organ + 1):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index,  :, :] = temp_target.reshape(temp_target.shape[0], imsize, imsize)

        organ_target = organ_target.cuda()
            # loss
        dice_0=0

        dice_stage1 = 0.0   
        smooth = 1.
        
        for organ_index in  range(num_organ + 1):
            pred_tep=pred_stage1[:, organ_index,  :, :] 
            target_tep=organ_target[:, organ_index,  :, :]
            
            pred_tep=pred_tep.contiguous().view(-1)
            target_tep=target_tep.contiguous().view(-1)
            intersection_tp = (pred_tep * target_tep).sum()
            dice_tp=(2. * intersection_tp + smooth)/(pred_tep.sum() + target_tep.sum() + smooth)
            
            if organ_index==0:
                dice_0=dice_tp
            
        return dice_0

    def set_test_input(self,input):
        self.test_A=input#torch.split(input_A1, input_A1.size(0), dim=1)    
     # def set_test_input(self,input,input_label):
     #    self.test_A,self.test_A_y=input, input_label#torch.split(input_A1, input_A1.size(0), dim=1)    
            

    def set_input(self, input):
        #AtoB = self.opt.which_direction == 'AtoB'  ## this is a misleading name -- a relic of using cycleGAN
        input_A1=input#[0]
        #input_A1=input_A1.view(-1,2,512,512)
        if(self.opt.model_type == 'multi'):
           input_A1=input_A1.view(-1,3,imsize,imsize)
           input_A11, input_A12, input_A13 = torch.split(input_A1.size(1)//3, dim=1)
           self.input_A.resize_(input_A11.size()).copy_(input_A11)
           self.input_A_y.resize_(input_A12.size()).copy_(input_A12)
           self.input_A_z.resize_(input_A13.size()).copy_(input_A13)
        else:
           input_A1=input_A1.view(-1,2,imsize,imsize)
           input_A11,input_A12=torch.split(input_A1, input_A1.size(1)//2, dim=1)
           self.input_A.resize_(input_A11.size()).copy_(input_A11)
           self.input_A_y.resize_(input_A12.size()).copy_(input_A12)
           
        self.image_paths = 'test'#input['A_paths' if AtoB else 'B_paths']

    def set_input_multi(self, input_x, input_y, input_z):
        AtoB = self.opt.which_direction == 'AtoB'
        
        input_A11, input_A12, input_A13 = input_x, input_y, input_z
        self.input_A.resize_(input_A11.size()).copy_(input_A11)
        self.input_A_y.resize_(input_A12.size()).copy_(input_A12)
        self.input_A_z.resize_(input_A13.size()).copy_(input_A13)
        self.image_paths = 'test'#input['A_paths' if AtoB else 'B_paths']
        
    def set_input_sep(self, input_x,input_y):
        AtoB = self.opt.which_direction == 'AtoB'

        input_A11,input_A12=input_x,input_y#torch.split(input_A1, input_A1.size(1)//2, dim=1)

        self.input_A.resize_(input_A11.size()).copy_(input_A11)


        self.input_A_y.resize_(input_A12.size()).copy_(input_A12)

        self.image_paths = 'test'#input['A_paths' if AtoB else 'B_paths']
    
    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_A_y=Variable(self.input_A_y)
        if(self.opt.model_type == 'multi'):
           self.class_A=Variable(self.input_A_z)
        
    
    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        self.seg_A = self.netSeg_A(real_A).data

    def net_Classtest_image(self):
        
        self.test_A=self.test_A.cuda()
        self.test_A_y=self.test_A_y.cuda()
        test_img=self.test_A
        test_img = test_img.float()

        A_class = self.netSeg_A(test_img)
        A_class = torch.argmax(A_class, dim=1)
        A_class = A_class.view(1,1)
        A_class_out = A_class.data
        
        return self.test_A.cpu().float().numpy(),A_class_out.cpu().float().numpy()
    

    def net_Segtest_image(self):

        self.test_A=self.test_A.cuda()
        #self.test_A_y=self.test_A_y.cuda()
        test_img=self.test_A
        if(self.opt.model_type == 'deep'):
           _,A_AB_seg=self.netSeg_A(test_img)
        else:
           A_AB_seg=self.netSeg_A(test_img)
           
        #loss=self.dice_loss(A_AB_seg,self.test_A_y, test_img)
        
        A_AB_seg=F.softmax(A_AB_seg, dim=1)
        A_AB_seg=A_AB_seg[:,1,:,:]

        
        A_AB_seg=A_AB_seg.view(1,1, imsize, imsize)
        A_AB_seg_out=A_AB_seg.data

        A_AB_seg_out=A_AB_seg_out.view(1,1, imsize, imsize)

        #A_y_out=self.test_A_y.data
        #self.test_A_y=self.test_A_y.cuda()

        test_A_data=self.test_A.data
        A_AB_seg=A_AB_seg.data
        #A_y=self.test_A_y.data
        test_A_data,d999=self.tensor2im_jj(test_A_data)

        A_AB_seg=util.tensor2im_scaled(A_AB_seg)
        #A_y=util.tensor2im_scaled(A_y)

        test_A_data=test_A_data[:,imsize:imsize*2,:]

        A_AB_seg=A_AB_seg#[:,256:512,:]
        #A_y=A_y#[:,256:512,:]

        
        #image_numpy_all=np.concatenate((test_A_data,A_y,),axis=1)
        #image_numpy_all=np.concatenate((image_numpy_all,A_AB_seg,),axis=1)


        #return loss,self.test_A.cpu().float().numpy(),A_AB_seg_out.cpu().float().numpy(),A_y_out.cpu().float().numpy(),image_numpy_all
        #return self.test_A.cpu().float().numpy(),A_AB_seg_out.cpu().float().numpy(),image_numpy_all
        return self.test_A.cpu().float().numpy(),A_AB_seg_out.cpu().float().numpy()
    
    def get_image_paths(self):
        return self.image_paths
    
    def cal_seg_loss (self,netSeg,pred,gt):
        img=pred
        lmd = 1
        if(self.opt.model_type == 'deep'):
           out1, self.pred = netSeg(pred)
           seg_loss = lmd*(self.dice_loss(self.pred, gt, img)*self.opt.out_wt + (1. - self.opt.out_wt)*self.dice_loss(out1, gt, img))
        else:  
           #print('Input Shape Unet: ', self.pred.shape())
           self.pred=netSeg(pred)
           seg_loss=lmd*self.dice_loss(self.pred,gt,img)
        return seg_loss


    def backward_Seg_A(self):
        gt_A=self.real_A_y # gt 
        img_A=self.real_A # gt
        
        seg_loss = self.cal_seg_loss(self.netSeg_A, img_A, gt_A)
        if(self.opt.model_type == 'deep'):
           _,out = self.netSeg_A(img_A)
           d0 = self.cal_dice_loss(out, gt_A)
        else:
           d0 = self.cal_dice_loss(self.netSeg_A(img_A), gt_A)
        self.d0 = d0.item()
        self.seg_loss = seg_loss
        seg_loss.backward()

    def load_MR_seg_A(self, weight):
        self.load_network(self.netSeg_A,'Seg_A',weight)

    def optimize_parameters(self):
        # forward
        self.forward()
        self.optimizer_Seg_A.zero_grad()

        self.backward_Seg_A()
        self.optimizer_Seg_A.step()


    def get_current_errors(self):
        ret_errors = OrderedDict([('Seg_loss',  self.seg_loss), ('d0', self.d0)])
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        real_Ay=util.tensor2im_scaled(self.input_A_y)
        if(self.opt.model_type == 'deep'):
          _,pred_A = self.netSeg_A(self.input_A)
        else:
          pred_A=self.netSeg_A(self.input_A)
        pred_A=F.softmax(pred_A, dim=1)

        pred_A=torch.argmax(pred_A, dim=1)
        pred_A=pred_A.view(self.input_A.size()[0],1,imsize, imsize)
        pred_A=pred_A.data

        seg_A=util.tensor2im_scaled(pred_A) #

        ret_visuals = OrderedDict([('real_A', real_A),('real_A_GT_seg',real_Ay),('real_A_seg', seg_A)])
        return ret_visuals

    def get_current_seg(self):
        ret_visuals = OrderedDict([('d0', self.d0),])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netSeg_A, 'Seg_A', label, self.gpu_ids)
    
    def tensor2im_jj(self,image_tensor):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy_tep=image_numpy
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        elif (image_numpy.shape[0] != 3):
            k=floor(image_numpy.shape[0]/(2))
            image_numpy = np.tile(image_numpy[k,:,:], (3, 1, 1))

        self.test_A_tep = self.test_A[0].cpu().float().numpy()
        if self.test_A_tep.shape[0] == 1:
            self.test_A_tep = np.tile(self.test_A_tep, (3, 1, 1))
        elif (self.test_A_tep.shape[0] != 3):
            k=floor(self.test_A_tep.shape[0]/(2))
            self.test_A_tep = np.tile(self.test_A_tep[k,:,:], (3, 1, 1))
            
        image_numpy_all=np.concatenate((self.test_A_tep,image_numpy,),axis=2)
        image_numpy_all = (np.transpose(image_numpy_all, (1, 2, 0)) + 1) / 2.0 * 255.0    

        return image_numpy_all.astype(np.uint8),image_numpy_tep

    def tensor2im_jj_3(self,image_tensor):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy_tep=image_numpy
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        self.test_A_tep = self.test_A[0].cpu().float().numpy()
        if self.test_A_tep.shape[0] == 1:
            self.test_A_tep = np.tile(self.test_A_tep, (3, 1, 1))

        image_numpy_all=np.concatenate((self.test_A_tep,image_numpy,),axis=2)
        image_numpy_all = (np.transpose(image_numpy_all, (1, 2, 0)) + 1) / 2.0 * 255.0        
        

        return image_numpy_all.astype(np.uint8),image_numpy_tep
