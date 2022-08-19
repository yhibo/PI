
import SimpleITK as sitk
from strain_from_motion import *
from utils.utils_aha import *
import constant

def get_strain(dfield, myo):
    #myo = sitk.GetImageFromArray(mask[0,:,:,:])
    coord_i, data_i = lv_local_coord_system(myo, 0, False)

    Icoord_i = sitk.GetArrayFromImage(coord_i)

    #  # coord --> (dim, 9): # 3x3 = [c_l,c_c,c_r]
    Icoord_i = Icoord_i.reshape(Icoord_i.shape[:3]+(3,3))
    ldir = Icoord_i[...,0]
    cdir = Icoord_i[...,1]
    rdir = Icoord_i[...,2]
    cooord = [ldir, cdir, rdir]

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
        df = dfield[t].transpose((3,0,1,2))
        mk = sitk.GetArrayFromImage(myo)
        (iec[t], ier[t], ierc[t], ec[t], er[t], erc[t],
        Ec[t], Er[t], Erc[t]) = cine_dense_strain2D(df=df, Icoord=Icoord[0], mask=mk, ba_channel=0)
        (iecm[t], ierm[t], iercm[t], ecm[t], erm[t], ercm[t],
        Ecm[t], Erm[t], Ercm[t]) = (iec[t][mk==label].mean(), ier[t][mk==label].mean(), ierc[t][mk==label].mean(), 
                                                        ec[t][mk==label].mean(),er[t][mk==label].mean(), erc[t][mk==label].mean(),
                                                        Ec[t][mk==label].mean(), Er[t][mk==label].mean(), Erc[t][mk==label].mean())
        strain[t, 0] = erm[t]
        strain[t, 1] = ecm[t]
    
    return strain
