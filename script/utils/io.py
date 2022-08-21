"""
File: io.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://github.com/curiale
Description: Funciones auxiliares de IO
"""

import os
import numpy as np
import SimpleITK as sitk
from natsort import natsorted
import nibabel as nib


def read_mhd_to_nifti(data_folder, patient):
    images = [
        sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_folder, f)))
        for f in natsorted(os.listdir(data_folder))
        if f.endswith(".mhd") and f.startswith(f"{patient}_cSAX_time")
    ]

    return nib.Nifti1Image(np.array(images).transpose(), None)

def convert_mhd_to_nifti(data_folder, patient, out_folder):
    out_file = os.path.join(out_folder, f"{patient}.nii.gz")
    csax_images = (
        [
            sitk.ReadImage(os.path.join(data_folder, f))
            for f in natsorted(os.listdir(data_folder))
            if f.endswith(".mhd") and f.startswith(f"{patient}_cSAX_time")
        ]
    )
    print(out_file)
    sitk.WriteImage(sitk.JoinSeries(csax_images), out_file)