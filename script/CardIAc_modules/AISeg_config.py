'''
Author: Lucca Dellazoppa
Institution: Instituto Balseiro
Contact: lucca.dellazoppa.m@gmail.com
Description: file for loading AI-Segmentation's module configurations.
'''
import os, platform
from pathlib import Path

# CNN paths 
base_path = os.path.realpath(__file__).split(os.path.basename(os.path.realpath(__file__)))[0]
base_path = Path(base_path).parent
CNNsFolder = "CNNs"
ModelCenter = "vnet_rlo_2Dcenter.h5"
ModelSegm = "GVUNet.h5"

cnn_center_path = base_path / CNNsFolder / ModelCenter
cnn_segm_path = base_path / CNNsFolder / ModelSegm

# Tissues labels
RV = 1
myo = 2
LV = 3
label_tissues = {'RV':RV,'myo':myo,'LV':LV,'vector':[RV,myo,LV]}

# Density of myo
myo_density = 1.05*10**-3 # [g/mm3] 

# Logic initial conditions
logic_IC = {
    'segmStatus':-1,
    'backgName_default':'proxy_Backg_cardIAc',
    'labelName_default':'proxy_Label_cardIAc',
    'backgNameShort': 'proxy_Backg_Interval_cardIAc',
    'cnn_input_label':'unet_input',
    'cnn_output_label':'gvae_output',
    'cnn_roi_size':90
    }