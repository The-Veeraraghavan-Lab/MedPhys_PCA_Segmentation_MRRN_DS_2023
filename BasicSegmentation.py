# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 12:18:47 2023

@author: SimethJ
"""
import torch.multiprocessing
import h5py
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import scipy.io as sio
import os
import torch.utils.data
from options.seg_options import SegmentationOptions
from models.models import create_model
from util import util
import numpy as np


opt = SegmentationOptions().parse()
opt.isTrain = False

class Custom_data2(Dataset):

    def __init__(self, data_folder, img_list):

        self.data_path = data_folder
        self.img_file_list = []
        with open(img_list) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                self.img_file_list.append(line)

    def __len__(self):
        return len(self.img_file_list)

    def __getitem__(self, idx):
        keyname = 'img'
        if idx >= len(self.img_file_list):
            raise StopIteration

        img_idx = self.img_file_list[idx]
        mdata = h5py.File(self.data_path+img_idx+'.mat', 'r')
        img_raw_all = mdata[keyname]
        img = img_raw_all[:, :, :]
        img = np.float32(img)
        img = normalize_data_JJS(img)
        return img, img_idx
        
    
def normalize_data_JJS(data):
    # Assumes ADC values in 
    data[data < 0] = 0
    data[data > 3500] = 3500  # 1524
    data = data*2./3500 - 1  # 1500 - 1
    return (data)


path=os.getcwd()
model = create_model(opt) 
model_path = path+'/deep_model/MRRNDS_model' #load weights

model.load_MR_seg_A(model_path) #use weights

#Prep model for eval
for m in model.netSeg_A.modules():
    for child in m.children():
        if type(child) == torch.nn.BatchNorm2d:
            child.track_running_stats = False
            child.running_mean = None
            child.running_var = None
        
model.netSeg_A.eval()
    
#Load Data
data_path = path+'/Example_Data/'
data_list = path+'/Example_List.txt' #list of all image files to process
data_set = Custom_data2(data_path, data_list)

#Data Loader
loader1 = torch.utils.data.DataLoader(data_set,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)

#save location
save_path = path+'/segmentation_maps/'
util.mkdirs(save_path)


## Execute Segmentation + Save Segmentations
print('Running segmentation')
for i, (data_val,fname) in enumerate(loader1):
        fname = fname[0]
        model.set_test_input(data_val)
        ori_img, seg = model.net_Segtest_image()


        seg = np.array(seg)
        seg=np.squeeze(seg)
        seg = (seg > 0.5)
        seg = np.float32(seg)
        
        save_name = save_path+'%s.mat' % (fname)
        mat_dict = {'seg': seg}
        sio.savemat(save_name, {'Image_data': mat_dict})

print('Done! images saved to ' + save_path)


