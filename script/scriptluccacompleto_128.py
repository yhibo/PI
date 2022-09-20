from seglucca import get_segmentation
from motion_and_strain_128 import get_motion_and_strain
from utils.io import convert_mhd_to_nifti
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from datasets.base_dataset import pad_256x256
import time

patients = [f"v{p}" for p in range(1, 17) if p != 3]
output_folder = os.path.join(os.getcwd(),'results')

for patient in patients:
    print(patient)
    data_folder = os.path.join(os.getcwd(),'MHD_Data',patient,'cSAX')
    print("Cargo volumen")
    ####### TRANSFORMO MHD EN NIFTI
    volume_folder = os.path.join(output_folder,'images','volume')
    convert_mhd_to_nifti(data_folder, patient, volume_folder)
    # Me lo tira en x,y,z
    cine = nib.load(os.path.join(volume_folder, f"{patient}.nii.gz"))
    if(np.shape(cine)[0:2]!=(256,256)):
        print("Slice no es de 256x256, corrigiendo")
        data = cine.get_fdata()
        data = pad_256x256(data)
        cine = nib.Nifti1Image(data, cine.affine, cine.header)
    print("Shape de cine: ", np.shape(cine))
        

    print("Segmento")
    start_time = time.time()
    ####### SEGMENTACION
    seg_folder = os.path.join(output_folder,'images','our_seg')
    myo = nib.nifti1.Nifti1Image(get_segmentation(data_folder, patient), cine.affine)
    nib.save(myo, os.path.join(seg_folder, f"{patient}_seg.nii.gz"))
    print("Shape de segmentacion: ", np.shape(myo))
    print("--- Segmentation took %s seconds ---" % (time.time() - start_time))


    print("Deformacion")
    start_time = time.time()
    ####### MOVIMIENTO Y STRAIN
    dfield, strain, iec_aha, ier_aha, ierc_aha, iel_aha, seg_aha = get_motion_and_strain(cine, myo)
    print("--- Motion estrimation took %s seconds ---" % (time.time() - start_time))

    aha_img = nib.Nifti1Image(seg_aha, cine.affine)
    nib.save(aha_img, os.path.join(output_folder,'images','our_seg',f"{patient}_aha.nii.gz"))
    print("Shape de aha: ", np.shape(aha_img))

    dfield_folder = os.path.join(output_folder,'dfield','our_seg')
    np.save(os.path.join(dfield_folder, f"{patient}_dfield_reentrenado_128.npy"), dfield)
    print("Shape de dfield: ", np.shape(dfield))

    strain_folder = os.path.join(output_folder,'strain','our_seg')
    strain_aha = np.asarray([iec_aha, ier_aha, ierc_aha, iel_aha])
    np.save(os.path.join(strain_folder, f"{patient}_strain_aha_reentrenado.npy"), strain_aha)
    np.save(os.path.join(strain_folder, f"{patient}_strain_reentranado.npy"), np.asarray(strain))
