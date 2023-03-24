Author: Josiah Simeth, adapted from Jue jiang's code
Prostate DIL autosegmentation model from https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.16320

Example use found in "BasicSegmentation.py". Running "BasicSegmentation.py" will take the ADC image files in the folder "Example_Data" listed in "Example_List.txt." and produce segmenations using the model in "deep_model" and save the segmentations to "segmentation_maps"

Model weights found at https://www.dropbox.com/s/hz39lzewut5o320/MRRNDS_model_net_Seg_A.pth?dl=0

Input data are 128x128x5 (0.625x0.625x3 mm^3) ADC images centered on the prostate (example code takes .mat files with data saved to 'img' keyword). The model returns the segmentation for the central slice.
