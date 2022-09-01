import SimpleITK as sitk
from strain_from_motion import *
from utils.utils_aha import *
import nibabel as nib
import os
import constant
from datasets.base_dataset import _roll2center_crop
from datasets.base_dataset import _roll2center
from datasets.nifti_dataset import resample_nifti
from scipy.ndimage.measurements import center_of_mass
import pandas as pd
import itertools
import matplotlib.pyplot as plt

patient = 'v9'

mask = sitk.GetArrayFromImage(sitk.ReadImage("v9/cMAC/GT/SSFP/GT_Corrected/v9_corr.mhd"))

dfi = [None]*constant.nframes
dfi[0] = sitk.GetArrayFromImage(sitk.ReadImage(f"v9/MHD_Data/v9/DField/cSAX_222_221/v9_cSAX_dfield_3d-edge_ft_demons_time_1.mhd"))
for t in range(1, constant.nframes):
    dfi[t] = dfi[t-1] + sitk.GetArrayFromImage(sitk.ReadImage(f"v9/MHD_Data/v9/DField/cSAX_222_221/v9_cSAX_dfield_3d-edge_ft_demons_time_{t+1}.mhd"))

print(patient)

#I = np.argmax((mask == 1).sum(axis=(1, 2)))
#if I > mask.shape[0] // 2:
#    mask = mask[ ::-1, :, :]

Icoord = []

myo = sitk.GetImageFromArray(mask)
coord_i, data_i = lv_local_coord_system(myo, 0, False)

Icoord_i = sitk.GetArrayFromImage(coord_i)

#  # coord --> (dim, 9): # 3x3 = [c_l,c_c,c_r]
Icoord_i = Icoord_i.reshape(Icoord_i.shape[:3]+(3,3))
ldir = Icoord_i[...,0]
cdir = Icoord_i[...,1]
rdir = Icoord_i[...,2]
Icoord.append([ldir, cdir, rdir])

iec, ier, ierc, ec, er, erc, Ec, Er, Erc, iel, el, El = ([None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes, 
                                                        [None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes, 
                                                        [None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes,
                                                        [None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes)

iecm, ierm, iercm, ecm, erm, ercm, Ecm, Erm, Ercm, ielm, elm, Elm = ([None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes, 
                                                        [None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes, 
                                                        [None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes,
                                                        [None]*constant.nframes, [None]*constant.nframes, [None]*constant.nframes)


label = 1
strain = np.zeros((constant.nframes, 2))

for t in range(constant.nframes):
    #dfi.append(_roll2center(x=dfield[t].squeeze(), center=center).transpose((3,2,0,1))[(1,0,2),:,:,:])
    df = dfi[t].transpose((3,0,1,2))
    mk = mask
    (iec[t], ier[t], ierc[t], ec[t], er[t], erc[t],
    Ec[t], Er[t], Erc[t]) = cine_dense_strain2D(df=df, Icoord=Icoord[0], mask=mk, ba_channel=0)
    (iecm[t], ierm[t], iercm[t], ecm[t], erm[t], ercm[t],
    Ecm[t], Erm[t], Ercm[t]) = (iec[t][mk==label].mean(), ier[t][mk==label].mean(), ierc[t][mk==label].mean(), 
                                                    ec[t][mk==label].mean(),er[t][mk==label].mean(), erc[t][mk==label].mean(),
                                                    Ec[t][mk==label].mean(), Er[t][mk==label].mean(), Erc[t][mk==label].mean())

strain[:, 0] = np.array(erm) - np.arange(constant.nframes) * np.array(erm) / ( constant.nframes - 1)
strain[:, 1] = np.array(ecm) - np.arange(constant.nframes) * np.array(ecm) / ( constant.nframes - 1)
np.save("strain_nuestro/our_seg/strain_demons_{}.npy".format(patient), strain)


plt.figure()
#strainellos = np.load(os.path.join(strainellosfolder, f"strain_{patient}.npy"))
plt.plot(iecm-iecm[-1]*np.arange(constant.nframes)/(constant.nframes-1), color='c', label="C-nuestro", marker='.')
plt.plot(ierm-ierm[-1]*np.arange(constant.nframes)/(constant.nframes-1), color='r', label="R-nuestro", marker='.')
#plt.plot(strainellos[:,1], color='c', label="C-ellos", )
#plt.plot(strainellos[:,0], color='r', label="R-ellos")
plt.legend()
plt.show()