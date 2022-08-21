from models import deep_strain_model
from datasets import base_dataset
from datasets.nifti_dataset import resample_nifti
from tensorflow.keras.optimizers import Adam
from scipy.ndimage.measurements import center_of_mass
from datasets.base_dataset import roll_and_pad_256x256_to_center_inv
from datasets.base_dataset import _roll2center_crop
from scipy.ndimage import gaussian_filter
import numpy as np
import SimpleITK as sitk
from utils.strain_from_motion import *
from utils.utils_aha import *
import constant

##########################      Normalization     ######################################


def normalize(x, axis=(0, 1, 2)):
    # normalize per volume (x,y,z) frame
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / (sd + 1e-8)

######################### Constants and arguments ######################################

class CarMEN_options:
    def __init__(self):
        self.isTrain = False
        self.volume_shape = (128, 128, 16, 1)
        self.pretrained_models_netS = "models/carson_Jan2021.h5"
        self.pretrained_models_netME = "models/carmen_Jan2021.h5"


#########################     Motion and Strain   ######################################

def get_motion_and_strain(V_nifti, M_nifti):

    opt = CarMEN_options()
    model = deep_strain_model.DeepStrain(Adam, opt=opt)
    netME = model.get_netME()

    print("Motion and strain on patient.")

    V_nifti_resampled = resample_nifti(
        V_nifti, order=1, in_plane_resolution_mm=1.25, number_of_slices=16
    )
    M_nifti_resampled = resample_nifti(
        M_nifti, order=0, in_plane_resolution_mm=1.25, number_of_slices=16
    )

    center = center_of_mass(M_nifti_resampled.get_fdata()[:, :, :, 0] == 2)
    V = V_nifti_resampled.get_fdata()
    M = M_nifti_resampled.get_fdata()

    rv_pt = center_of_mass(M[:, :, :, 0] == 1)

    V = _roll2center_crop(x=V, center=center)

    I = np.argmax((M == 1).sum(axis=(0, 1, 3)))
    if I > M.shape[2] // 2:
        print("flip")
        V = V[:, :, ::-1]
        M = M[:, :, ::-1]

    V = normalize(V, axis=(0, 1, 2))

    mask = M.transpose((3,2,1,0))
    mask = mask==2
    mask = mask.astype(float)

    myo = sitk.GetImageFromArray(mask[0,:,:,:])
    coord_i, data_i = lv_local_coord_system(myo, 0, True)
    aha_img = create_aha(myo, rv_pt, 0, True)

    Icoord_i = sitk.GetArrayFromImage(coord_i)
    seg_aha = sitk.GetArrayFromImage(aha_img)

    #  # coord --> (dim, 9): # 3x3 = [c_l,c_c,c_r]
    Icoord_i = Icoord_i.reshape(Icoord_i.shape[:3]+(3,3))
    ldir = Icoord_i[...,0]
    cdir = Icoord_i[...,1]
    rdir = Icoord_i[...,2]
    Icoord = [ldir, cdir, rdir]

    dfield = []

    iec_aha = np.zeros((16, constant.nframes))
    ier_aha = np.zeros((16, constant.nframes))
    ierc_aha = np.zeros((16, constant.nframes))
    iel_aha = np.zeros((16, constant.nframes))

    iec, ier, ierc, ec, er, erc, Ec, Er, Erc, iel, el, El = ([None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes, 
                                                            [None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes, 
                                                            [None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes,
                                                            [None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes)

    iecm, ierm, iercm, ecm, erm, ercm, Ecm, Erm, Ercm, ielm, elm, Elm = ([None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes, 
                                                            [None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes, 
                                                            [None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes,
                                                            [None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes)
    
    label = 1
    strain = np.zeros((constant.nframes, 3))

    for t in range(constant.nframes):
        V_0 = V[..., 0][None, ..., None]
        V_t = V[..., t][None, ..., None]
        df = gaussian_filter(netME([V_0, V_t]).numpy(), sigma=(0,2,2,0,0)).squeeze()

        nifti_info = {'center_resampled' : center,
                    'center_resampled_256x256' : (128,128),
                    'shape_resampled' : (256,256,16)}
        df = roll_and_pad_256x256_to_center_inv(df, nifti_info)
        df = df.transpose((3,2,1,0))
        dfield.append(df)
        mk = mask[0,:,:,:]

        (iec[t], ier[t], ierc[t], ec[t], er[t], erc[t],
        Ec[t], Er[t], Erc[t], iel[t], el[t], El[t]) = cine_dense_strain3D(df=df, Icoord=Icoord, mask=mk, ba_channel=0)
        (iecm[t], ierm[t], iercm[t], ecm[t], erm[t], ercm[t],
        Ecm[t], Erm[t], Ercm[t], ielm[t], elm[t], Elm[t]) = (iec[t][mk==label].mean(), ier[t][mk==label].mean(), ierc[t][mk==label].mean(), 
                                                        ec[t][mk==label].mean(),er[t][mk==label].mean(), erc[t][mk==label].mean(),
                                                        Ec[t][mk==label].mean(), Er[t][mk==label].mean(), Erc[t][mk==label].mean(),
                                                        iel[t][mk==label].mean(), el[t][mk==label].mean(), El[t][mk==label].mean())
        strain[t, 0] = ierm[t]
        strain[t, 1] = iecm[t]
        strain[t, 2] = ielm[t]

        for j in range(16):
            rr,cc,jj = np.where(seg_aha == j+1)

            # rho is  1.05 g/cm^3 = 1.05*1e-3 g/mm^3
            #  vol_aha[j, i] = rr.size * rho
            iec_aha[j, t] = iec[t][rr,cc,jj].mean() * 100
            ier_aha[j, t] = ier[t][rr,cc,jj].mean() * 100
            ierc_aha[j, t] = ierc[t][rr,cc,jj].mean() * 100
            iel_aha[j, t] = iel[t][rr,cc,jj].mean() * 100
        

    return (np.asarray(dfield), strain, iec_aha, ier_aha, ierc_aha, iel_aha, seg_aha)