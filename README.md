# Deep learning-based dominant index lesion segmentation for MR-guided radiation therapy of prostate cancer
## Josiah Simeth, Jue Jiang, Anton Nosov, Andreas Wibmer, Michael Zelefsky, Neelam Tyagi, Harini Veeraraghavan

Here is the code for our accepted manuscript accepted to Medical Physics for generating segmentations of prostate cancers from apparent coefficient diffusion MRI images. Our method extends the multiple resolution residual network to extracts segmentation of the prosate cancers for application to hypofractionated MR guided radiotherapy of prostate cancers. 

The method accepts inputs 128x128x5 (0.625x0.625x3 mm^3) ADC images centered on the prostate (example code takes .mat files with data saved to 'img' keyword). The model returns the segmentation for the central slice.

Example use of the method is shown in "BasicSegmentation.py". Running "BasicSegmentation.py" will take the ADC image files in the folder "Example_Data" listed in "Example_List.txt." and produce segmenations using the model in "deep_model" and save the segmentations to "segmentation_maps"

Alternatively, just use PCA_inference to segment nii volumes in nii_vols folder (1 example ProstateX image included). nii volumes for each patient should be placed in seperate folders in nii_vols with "src" in the filename to designate it as the volume to segment and to distinguish it from segmentations or T1 images. 

Model weights to use this model are provided at https://www.dropbox.com/s/hz39lzewut5o320/MRRNDS_model_net_Seg_A.pth?dl=0

Please contact us if you have any questions regarding the use of this approach. 

If you do use our method, please cite our work that is found here:
Prostate DIL autosegmentation model from https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.16320

or in the arXiv: 
https://arxiv.org/abs/2303.03494
