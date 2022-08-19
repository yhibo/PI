from seg import get_segmentation
from motion_and_strain import get_motion_and_strain
from utils.io import read_mhd_to_nifti
import os
import numpy as np
import matplotlib.pyplot as plt

patient = 'v9'
data_folder = os.path.join(os.getcwd(),'MHD_Data',patient,'cSAX')

####### TRANSFORMO MHD EN NIFTI
# Me lo tira en x,y,z
cine = read_mhd_to_nifti(data_folder, patient)

####### SEGMENTACION
####### RECIBE CINE EN NIFTI
myo = get_segmentation(cine)
####### DEVUELVE SEG EN NIFTI

####### MOVIMIENTO Y STRAIN
dfield, strain = get_motion_and_strain(cine, myo)

plt.figure()
plt.title('SEG DEEP Y CAMPO DEEP')
plt.plot(strain[:,1], color='c', label="C", marker='.')
plt.legend()
plt.figure()
plt.title('SEG DEEP Y CAMPO DEEP')
plt.plot(strain[:,0], color='r', label="R", marker='.')
plt.legend()
plt.show()


