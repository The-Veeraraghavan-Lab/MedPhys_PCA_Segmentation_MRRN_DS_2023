import torch.nn as nn
import torch as t



class Residual_Unit(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, in_c,inter_c,out_c):
        super(Residual_Unit, self).__init__()
        
        self.unit=CNN_block(in_c,inter_c)

    def forward(self, x):
        x_=self.unit(x)

        return x+x_

class CNN_block(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, in_c,inter_c):
        super(CNN_block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_c, inter_c, 3, 1, padding=1,bias=True)
        self.norm1=nn.BatchNorm2d(inter_c)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x1=self.conv1(x)
        x2=self.norm1(x1)
        x3=self.activation(x2)

        return x3


class FRRU(nn.Module):
    """FRRU for the MRRN net."""
    def __init__(self, in_c,inter_c,up_scale,adjust_channel,max_p_size):
        super(FRRU, self).__init__()
        
        self.maxp = nn.MaxPool2d(kernel_size=(max_p_size, max_p_size))
        self.drop=nn.Dropout2d(p=0.5)
        self.cnn_block=nn.Sequential(
            nn.Conv2d(in_c, inter_c, 3, 1, padding=1,bias=True),
            nn.BatchNorm2d(inter_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_c, inter_c, 3, 1,padding=1, bias=True),
            nn.BatchNorm2d(inter_c),
            nn.ReLU(inplace=True))
        self.channel_adjust=nn.Conv2d(inter_c, adjust_channel, 1, 1, padding=0,bias=True)
        self.upsample= nn.Upsample(scale_factor=up_scale,mode = "bilinear",align_corners=True)
        


    def forward(self, p_s,r_s):
        #pooling for residual stream
        r_s1=self.maxp(r_s)
        r_s1=self.drop(r_s1)
        #Apply drop out (could be turn off)

        merged_=t.cat((r_s1,p_s),dim=1)
        pool_sm_out=self.cnn_block(merged_)
        #merged feature processed for FRRU pooling stream output 
        
        #Next for residual output for FRRU residual stream output (pull back to residual stream)
        adjust_out1=self.channel_adjust(pool_sm_out)
        adjust_out1_up_samp=self.upsample(adjust_out1)
        residual_sm_out=adjust_out1_up_samp+r_s

        return pool_sm_out,residual_sm_out

class Incre_MRRN_deepsup(nn.Module):
    def __init__(self, n_channels, n_classes, deeplayer=0):
        super(Incre_MRRN_deepsup, self).__init__()

        self.deeplayer=deeplayer
        
        self.CNN_block1=CNN_block(n_channels,32)
        self.CNN_block2=CNN_block(96,32)
        self.CNN_block3=CNN_block(64,32)
        self.RU1=Residual_Unit(32,32,32)
        self.RU2=Residual_Unit(32,32,32)
        self.RU3=Residual_Unit(32,32,32)

        self.RU11=Residual_Unit(32,32,32)
        self.RU22=Residual_Unit(32,32,32)
        self.RU33=Residual_Unit(32,32,32)


        self.Pool_stream1=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Pool_stream2=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Pool_stream3=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Pool_stream4=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Up_stream1=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear",align_corners=True))
            #nn.Dropout2d(p=0.5))

        self.Up_stream2=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear",align_corners=True))
            #nn.Dropout2d(p=0.5))

        self.Up_stream3=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear",align_corners=True))
            #nn.Dropout2d(p=0.5))

        self.Up_stream4=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear",align_corners=True))
            #nn.Dropout2d(p=0.5))


        self.Residual_stream1 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        self.FRRU1_1=FRRU(64,64,2,32,2)
        self.FRRU1_2=FRRU(96,64,2,32,2)
        self.FRRU1_3=FRRU(96,64,2,32,2)

        self.FRRU2_1=FRRU(96,128,4,32,4)
        self.FRRU2_2=FRRU(192,128,2,64,2)
        self.FRRU2_3=FRRU(160,128,4,32,4)

        self.FRRU3_1=FRRU(160,256,8,32,8)
        self.FRRU3_2=FRRU(384,256,2,128,2)
        self.FRRU3_3=FRRU(320,256,4,64,4)
        self.FRRU3_4=FRRU(288,256,8,32,8)

        self.FRRU4_1=FRRU(288,512,16,32,16)
        self.FRRU4_2=FRRU(768,512,2,256,2)
        self.FRRU4_3=FRRU(640,512,4,128,4)
        self.FRRU4_4=FRRU(576,512,8,64,8)
        self.FRRU4_5=FRRU(544,512,16,32,16)

        self.FRRU33_1=FRRU(544,256,8,32,8)
        self.FRRU33_2=FRRU(384,256,2,128,2)
        self.FRRU33_3=FRRU(320,256,4,64,4)
        self.FRRU33_4=FRRU(288,256,8,32,8)

        self.FRRU22_1=FRRU(288,128,4,32,4)
        self.FRRU22_2=FRRU(192,128,2,64,2)
        self.FRRU22_3=FRRU(160,128,4,32,4)

        self.FRRU11_1=FRRU(96,64,2,32,2)
        self.FRRU11_2=FRRU(96,64,2,32,2)
        self.FRRU11_3=FRRU(96,64,2,32,2)

        self.out_conv=nn.Conv2d(32, 2, 1, 1, bias=True)
        self.out_conv1=nn.Conv2d(32, 2, 1, 1, bias=True)
        
        
        ## uncoment for deep 3
        #self.out_conv2=nn.Conv2d(32, 2, 1, 1, bias=True)
        
        #self.out_act=nn.Sigmoid()
        if self.deeplayer==4:
            self.deepsupconv=CNN_block(512,32)
        elif self.deeplayer==3:
            self.deepsupconv=CNN_block(256,32)
        elif self.deeplayer==2:
            self.deepsupconv=CNN_block(128,32)
        elif self.deeplayer==1:
            self.deepsupconv=CNN_block(64,32)
            
        
        
        
        
        
    def forward (self,x):

        x1= self.CNN_block1(x)   
        x2= self.RU1(x1)
        x3= self.RU2(x2)
        x4= self.RU3(x3)
        # Three RU 
        
        rs_2=self.Pool_stream1(x4) #Residual stream 2nd
        rs_1=self.Residual_stream1 (x4) #Residual stream 1st

        rs_2,rs_1=self.FRRU1_1(rs_2,rs_1)
        rs_2,rs_1=self.FRRU1_2(rs_2,rs_1)
        rs_2,rs_1=self.FRRU1_3(rs_2,rs_1)
        #print ('after FRRU1_3 ',rs_2.size())
        #print ('after FRRU1_3 ',rs_1.size())
        rs_3= self.Pool_stream2(rs_2)

        rs_3,rs_1=self.FRRU2_1(rs_3,rs_1)
        rs_3,rs_2=self.FRRU2_2(rs_3,rs_2)
        rs_3,rs_1=self.FRRU2_3(rs_3,rs_1)

        rs_4= self.Pool_stream3(rs_3)
        rs_4,rs_1=self.FRRU3_1(rs_4,rs_1)
        rs_4,rs_3=self.FRRU3_2(rs_4,rs_3)
        rs_4,rs_2=self.FRRU3_3(rs_4,rs_2)
        rs_4,rs_1=self.FRRU3_4(rs_4,rs_1)

        rs_5= self.Pool_stream4(rs_4)

        rs_5,rs_1=self.FRRU4_1(rs_5,rs_1)
        rs_5,rs_4=self.FRRU4_2(rs_5,rs_4)
        rs_5,rs_3=self.FRRU4_3(rs_5,rs_3)
        rs_5,rs_2=self.FRRU4_4(rs_5,rs_2)
        rs_5,rs_1=self.FRRU4_5(rs_5,rs_1)

        #Start to do the up-sampling pool
        rs_5=self.Up_stream1(rs_5)
        rs_4,rs_1=self.FRRU33_1(rs_5,rs_1)
        rs_4,rs_3=self.FRRU33_2(rs_4,rs_3)
        rs_4,rs_2=self.FRRU33_3(rs_4,rs_2)
        rs_4,rs_1=self.FRRU33_4(rs_4,rs_1)

        rs_4=self.Up_stream2(rs_4)
        rs_3,rs_1=self.FRRU22_1(rs_4,rs_1)
        rs_3,rs_2=self.FRRU22_2(rs_3,rs_2)
        rs_3,rs_1=self.FRRU22_3(rs_3,rs_1)

        rs_3=self.Up_stream3(rs_3)
        rs_2,rs_1=self.FRRU11_1(rs_2,rs_1)
        rs_2,rs_1=self.FRRU11_2(rs_2,rs_1)
        rs_2,rs_1=self.FRRU11_3(rs_2,rs_1)

        rs_2=self.Up_stream1(rs_2)
        ## when using deep 3 turn the below on
        #rs_21 = self.CNN_block3(rs_2)
        #out3 = self.out_conv2(rs_21)
        rs_1=t.cat((rs_1,rs_2),dim=1)
        rs_1=self.CNN_block2(rs_1)
        
        if self.deeplayer==4:
            rs_5=self.Up_stream1(rs_5)
            rs_5=self.Up_stream2(rs_5)
            rs_5=self.Up_stream3(rs_5)
            out2=self.deepsupconv(rs_5)
        elif self.deeplayer==3:
            rs_4=self.Up_stream2(rs_4)
            rs_4=self.Up_stream3(rs_4)
            out2=self.deepsupconv(rs_4)
        elif self.deeplayer==2:
            rs_3=self.Up_stream3(rs_3)
            out2=self.deepsupconv(rs_3)
        elif self.deeplayer==1:
            #print ('rs_2 size: ',rs_2.size())
            out2=self.deepsupconv(rs_2)
        else:
            out2=rs_1
        
        
        out2 = self.out_conv1(out2)
        
        rs_1= self.RU11(rs_1)
        rs_1= self.RU22(rs_1)
        rs_1= self.RU33(rs_1)
        #
        out=self.out_conv(rs_1)
        
        return out2, out#,rs_1, # out2, out3, out  #out2, out

