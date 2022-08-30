from seglucca import get_segmentation
from motion_and_strain import get_motion_and_strain
from utils.io import convert_mhd_to_nifti
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

patient = 'v9'
data_folder = os.path.join(os.getcwd(),'MHD_Data',patient,'cSAX')

print("Cargo volumen")
####### TRANSFORMO MHD EN NIFTI
convert_mhd_to_nifti(data_folder, patient, data_folder)
# Me lo tira en x,y,z
cine = nib.load(os.path.join(data_folder, f"{patient}.nii.gz"))

print("Segmento")
####### SEGMENTACION
myo = nib.nifti1.Nifti1Image(get_segmentation(data_folder, patient), cine.affine)
print(myo)

print("Deformacion")
####### MOVIMIENTO Y STRAIN
dfield, strain, iec_aha, ier_aha, ierc_aha, iel_aha, seg_aha = get_motion_and_strain(cine, myo)

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


