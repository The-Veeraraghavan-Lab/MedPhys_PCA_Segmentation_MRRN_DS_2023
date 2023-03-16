Author: Josiah Simeth, adapted from Jue jiang's code
Readme for prostate autosegmentation code for implementation in MIM. (https://github.mskcc.org/cerr-segmentation-containers/ADC_ProstDIL_MRRNDS)

Example use found in "BasicSegmentation.py". Running "BasicSegmentation.py" will take the ADC image files in the folder "Example_Data" listed in "Example_List.txt." and produce segmenations using the model in "deep_model" and save the segmentations to "segmentation_maps"

Model too large for github, copy of whole folder found in DeasyLab1\Josiah\For_MIM_Integration


External Dependencies:
conda                  4.14.0
conda-package-handling 1.7.2
h5py                   2.10.0
numpy                  1.19.2
pip                    20.3.1
scipy                  1.5.2
torch                  1.4.0