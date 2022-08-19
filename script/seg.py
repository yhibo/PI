from models import deep_strain_model
from datasets import base_dataset
from datasets.nifti_dataset import resample_nifti
from tensorflow.keras.optimizers import Adam
from scipy.ndimage.measurements import center_of_mass
import numpy as np

########################## Mask and normalization ######################################


def normalize(x, axis=(0, 1, 2)):
    # normalize per volume (x,y,z) frame
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / (sd + 1e-8)


def get_mask(V, netS):
    nx, ny, nz, nt = V.shape

    M = np.zeros((nx, ny, nz, nt))
    v = V.transpose((2, 3, 0, 1)).reshape((-1, nx, ny))  # (nz*nt,nx,ny)
    v = normalize(v)
    for t in range(nt):
        for z in range(nz):
            m = netS(
                v[z * nt + t, nx // 2 - 64 : nx // 2 + 64, ny // 2 - 64 : ny // 2 + 64][
                    None, ..., None
                ]
            )
            M[nx // 2 - 64 : nx // 2 + 64, ny // 2 - 64 : ny // 2 + 64, z, t] += (
                np.argmax(m, -1).transpose((1, 2, 0)).reshape((128, 128))
            )
    return M


######################### Constants and arguments ######################################

class CarSON_options:
    def __init__(self):
        self.isTrain = False
        self.image_shape = (128, 128, 1)
        self.nlabels = 4
        self.pretrained_models_netS = "models/carson_Jan2021.h5"
        self.pretrained_models_netME = "models/carmen_Jan2021.h5"


#########################       Segmentation      ######################################

def get_segmentation(V_nifti):

    opt = CarSON_options()
    model = deep_strain_model.DeepStrain(Adam, opt=opt)
    netS = model.get_netS()

    V_nifti_resampled = resample_nifti(
        V_nifti, order=1, in_plane_resolution_mm=1.25, number_of_slices=None
    )

    V = V_nifti_resampled.get_fdata()
    V = normalize(V, axis=(0, 1))
    M = get_mask(V, netS)

    center_resampled = center_of_mass(M[:, :, :, 0] == 2)
    V = base_dataset.roll_and_pad_256x256_to_center(x=V, center=center_resampled)
    M = base_dataset.roll_and_pad_256x256_to_center(x=M, center=center_resampled)
    center_resampled_256x256 = center_of_mass(M == 3)

    nifti_info = {
        "affine": V_nifti.affine,
        "affine_resampled": V_nifti_resampled.affine,
        "zooms": V_nifti.header.get_zooms(),
        "zooms_resampled": V_nifti_resampled.header.get_zooms(),
        "shape": V_nifti.shape,
        "shape_resampled": V_nifti_resampled.shape,
        "center_resampled": center_resampled,
        "center_resampled_256x256": center_resampled_256x256,
    }

    M = get_mask(V, netS)[128 - 64 : 128 + 64, 128 - 64 : 128 + 64]
    M_nifti = base_dataset.convert_back_to_nifti(
        M, nifti_info, inv_256x256=True, order=1, mode="nearest"
    )
    
    return M_nifti