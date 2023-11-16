# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:41:09 2023

@author: SimethJ
"""



# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:45:59 2023

@author: SimethJ

Updated inference function that will process all the nifti files in a specified
folder (default: )

assumes nifti files with units 1e-3 mm^2/s

"""

#import nibabel as nib
import numpy as np
#import matplotlib.pyplot as plt
import torchio as tio
import os

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
import torch as t
from pathlib import Path


opt = SegmentationOptions().parse()
opt.isTrain = False


def find_file_with_string(folder_path, target_string):
    # Get the list of files in the specified folder
    files = os.listdir(folder_path)
    # Iterate through the files and find the first file containing the target string
    for file in files:
        if target_string in file:
            # If the target string is found in the file name, return the file
            return file
    # If no matching file is found, return None
    return None


mr_paths = []
seg_paths = []

#Define input dimensions and resolution for inference model 
PD_in=np.array([0.6250, 0.6250, 3]) # millimeters
DIM_in=np.array([128,128,5]) # 128x128 5 Slices

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
                
path = path+'/nii_vols' #load weights


for directories in os.listdir(path): 
    #files=os.listdir(directories)
    path2=os.path.join(path,directories)
    
    #assumes keyword 'src_' in ADC filename
    mr_file=find_file_with_string(path2, 'src_')
    mr_path=os.path.join(path2, mr_file)
    
    
    print('segmenting %s' %mr_path)
    
    
    #seg = tio.LabelMap(seg_path)
    image = tio.ScalarImage(mr_path)
    
    adc_max=image.data.amax()
    adc_max=adc_max.data.cpu().numpy()
    
    if adc_max<0.05:# input likely in  mm^2/s, 
        print('Confirm units of ADC: input probably incorrectly in mm^2/s, normalizing with that assumption')
        multiplier=1e6
    elif adc_max<50:#input likely in  1e-3 mm^2/s, 
        print('Confirm units of ADC: input probably incorrectly in 1e-3  mm^2/s, normalizing with that assumption')
        multiplier=1
    elif adc_max<50000:#input likely in  1e-6 mm^2/s, 
        multiplier=1e-3
        #print('Confirm units of ADC: input probably incorrectly in 1e-6  mm^2/s, normalizing with that assumption')
    else:
        print('Confirm units of ADC: values too large to be 1e-6  mm^2/s')
    
    image.set_data(image.data)
    transform = tio.Resample(mr_path)

    input_resampler = tio.Resample((PD_in[0],PD_in[1], PD_in[2]))
    
    resampled_image=input_resampler(image)
    final_seg= tio.LabelMap(tensor=np.zeros(resampled_image.data.shape),affine=resampled_image.affine)
    
    _, _, _, z = np.meshgrid(1,range(resampled_image.data.shape[1]), range(resampled_image.data.shape[2]), range(resampled_image.data.shape[3]), indexing='ij')
    cropper_vol = tio.CropOrPad((DIM_in[0],DIM_in[1],resampled_image.data.shape[3]))
    final_seg=cropper_vol(final_seg)
    
    for i in range(resampled_image.data.shape[3]):
        slice_map = tio.LabelMap(tensor=(z==i),affine=resampled_image.affine)
        cropper = tio.CropOrPad((DIM_in[0],DIM_in[1],DIM_in[2]),mask_name='slice_map', padding_mode='reflect')
        
        subject=tio.Subject(im=resampled_image,slice_map=slice_map);
        crop_image=cropper(subject).im

        #set normalized data as input
        model.set_test_input((2*t.permute(crop_image.data*multiplier, (0,3, 1, 2))/3.5-1))
        ori_img, seg = model.net_Segtest_image()
        
        # assemble segmentation slcie by slice
        final_seg.data[0,:,:,i]=t.from_numpy(seg[0,0,:,:])
           
    uncrop=tio.CropOrPad((image.data.shape[1],image.data.shape[2],image.data.shape[3]))    
    output_resampler = tio.Resample(image)
    final_seg=uncrop(final_seg)
    final_seg=output_resampler(final_seg)
    
    
    
    #segmentation name, destination, and format decided here
    outfile= Path(mr_file).stem.replace('src', 'seg')   
    outpath = os.path.join(path2, outfile)
    final_seg.save(outpath + '.nii')    