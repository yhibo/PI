from seglucca import get_segmentation
from motion_and_strain import get_motion_and_strain
from utils.io import read_mhd_to_nifti
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

patient = 'v9'
data_folder = os.path.join(os.getcwd(),'MHD_Data',patient,'cSAX')

print("Cargo volumen")
####### TRANSFORMO MHD EN NIFTI
# Me lo tira en x,y,z
cine = read_mhd_to_nifti(data_folder, patient)

print("Segmento")
####### SEGMENTACION
myo = nib.nifti1.Nifti1Image(get_segmentation(data_folder, patient), np.eye(4))

print("Deformacion")
####### MOVIMIENTO Y STRAIN
dfield, strain = get_motion_and_strain(cine, myo)

plt.figure()
plt.title('SEG LUCCA Y CAMPO DEEP')
plt.plot(100*strain[:,1], color='c', label="C", marker='.')
plt.legend()
plt.figure()
plt.title('SEG LUCCA Y CAMPO DEEP')
plt.plot(100*strain[:,0], color='r', label="R", marker='.')
plt.legend()
plt.figure()
plt.title('SEG LUCCA Y CAMPO DEEP')
plt.plot(100*strain[:,2]*128/16, color='g', label="L", marker='.')
plt.legend()
plt.show()


