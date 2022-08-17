
### Prueba para strain2D con segmentación CARSON

patients = [f"v{p}" for p in range(9, 10) if p != 3]
dfields_folder = 'dfield/their_seg/'
images_folder = 'segs/their_seg/'
strainellosfolder = 'strain_ellos/their_seg/'
for patient in patients:
 
    print(patient)

    M_nifti = nib.load(os.path.join(images_folder, f"{patient}_seg.nii.gz"))
    M_nifti_resampled = resample_nifti(
                M_nifti, order=0, in_plane_resolution_mm=1.25
            )
    center = center_of_mass(M_nifti_resampled.get_fdata()[:, :, :, 0] == 2)
    M = M_nifti_resampled.get_fdata()
    M = _roll2center_crop(x=M, center=center)
    I = np.argmax((M == 1).sum(axis=(0, 1, 3)))
    if I > M.shape[2] // 2:
        M = M[:, :, ::-1]
    mask = M.transpose((3,2,0,1))
    mask = mask==2
    mask = mask.astype(float)

    Icoord = []

    for t in range(0, constant.nframes): 

        myo = sitk.GetImageFromArray(mask[t,:,:,:])
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

    dfield = np.load(os.path.join(dfields_folder, f"dfield_{patient}.npy"))

    #dfield = dfield[:,:,:,:,:,t2]
    
    label = 1
    strain = np.zeros((constant.nframes, 2))
    dfi = []

    for t in range(constant.nframes):
        #dfi.append(_roll2center(x=dfield[t].squeeze(), center=center).transpose((3,2,0,1))[(1,0,2),:,:,:])
        dfi.append(dfield[t].squeeze().transpose((3,2,0,1))[(1,0,2),:,:,:])
        df = dfi[t]
        mk = mask[t,:,:,:]
        (iec[t], ier[t], ierc[t], ec[t], er[t], erc[t],
        Ec[t], Er[t], Erc[t]) = cine_dense_strain2D(df=df, Icoord=Icoord[t], mask=mk, ba_channel=0)
        (iecm[t], ierm[t], iercm[t], ecm[t], erm[t], ercm[t],
        Ecm[t], Erm[t], Ercm[t]) = (iec[t][mk==label].mean(), ier[t][mk==label].mean(), ierc[t][mk==label].mean(), 
                                                        ec[t][mk==label].mean(),er[t][mk==label].mean(), erc[t][mk==label].mean(),
                                                        Ec[t][mk==label].mean(), Er[t][mk==label].mean(), Erc[t][mk==label].mean())
        strain[t, 0] = erm[t]
        strain[t, 1] = ecm[t]
    np.save("strain_nuestro/strain_{}.npy".format(patient), strain)



### Prueba para strain2D con segmentación NUESTRA

patients = [f"v{p}" for p in range(9, 10) if p != 3]
dfields_folder = 'dfield/our_seg/'
images_folder = 'segs/our_seg/'
strainellosfolder = 'strain_ellos/our_seg/'
for patient in patients:

    print(patient)

    M_nifti = nib.load(os.path.join(images_folder, f"{patient}_seg.nii.gz"))
    M_nifti_resampled = resample_nifti(
                M_nifti, order=0, in_plane_resolution_mm=1.25
            )
    center = center_of_mass(M_nifti_resampled.get_fdata()[:, :, :, 0] == 1)
    M = M_nifti_resampled.get_fdata()
    M = _roll2center_crop(x=M, center=center)
    I = np.argmax((M == 1).sum(axis=(0, 1, 3)))
    if I > M.shape[2] // 2:
        M = M[:, :, ::-1]
    mask = M.transpose((3,2,0,1))
    mask = mask==1
    mask = mask.astype(float)

    Icoord = []

    for t in range(0, constant.nframes): 

        myo = sitk.GetImageFromArray(mask[t,:,:,:])
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

    dfield = np.load(os.path.join(dfields_folder, f"dfield_{patient}.npy"))

    #dfield = dfield[:,:,:,:,:,t2]
    
    label = 1
    strain = np.zeros((constant.nframes, 2))
    dfi = []

    for t in range(constant.nframes):
        #dfi.append(_roll2center(x=dfield[t].squeeze(), center=center).transpose((3,2,0,1))[(1,0,2),:,:,:])
        dfi.append(dfield[t].squeeze().transpose((3,2,0,1))[(1,0,2),:,:,:])
        df = dfi[t]
        mk = mask[t,:,:,:]
        (iec[t], ier[t], ierc[t], ec[t], er[t], erc[t],
        Ec[t], Er[t], Erc[t]) = cine_dense_strain2D(df=df, Icoord=Icoord[t], mask=mk, ba_channel=0)
        (iecm[t], ierm[t], iercm[t], ecm[t], erm[t], ercm[t],
        Ecm[t], Erm[t], Ercm[t]) = (iec[t][mk==label].mean(), ier[t][mk==label].mean(), ierc[t][mk==label].mean(), 
                                                        ec[t][mk==label].mean(),er[t][mk==label].mean(), erc[t][mk==label].mean(),
                                                        Ec[t][mk==label].mean(), Er[t][mk==label].mean(), Erc[t][mk==label].mean())
        strain[t, 0] = erm[t]
        strain[t, 1] = ecm[t]
    np.save("strain_nuestro/strain_{}.npy".format(patient), strain)


#### Plot comparando el strain calculado por DeepStrain con nuestra metodología (usando un promedio global sobre el miocardio)    

plt.figure()
strainellos = np.load(os.path.join(strainellosfolder, f"strain_{patient}.npy"))
plt.plot(iecm, color='c', label="C-nuestro", marker='.')
plt.plot(ierm, color='r', label="R-nuestro", marker='.')
plt.plot(strainellos[:,1], color='c', label="C-ellos", )
plt.plot(strainellos[:,0], color='r', label="R-ellos")
plt.legend()

#### Plot para observar el strain calculado con nuestra metodología en el miocardio

from skimage.util import montage
frame = 10
plt.figure(figsize=(10,10))
plt.imshow(montage(np.array(ier)[frame,:,:,:]))
plt.colorbar()
#np.shape(np.array(ier)[:,5,:,:])

#### Visualización del strain calculado con Deepstrain superpuesto con la máscara del miocardio correspondiente

from skimage.util import montage
frame = 10
mask_rot = np.load("strain_ellos/our_seg/mask_rot_v9.npy")
straincompleto = np.load("strain_ellos/our_seg/completo_strain_v9.npy")
plt.figure(figsize=(10,10))
plt.imshow(montage((mask_rot[frame]==1).transpose((2,0,1))))
plt.imshow(montage((straincompleto[frame][0]).transpose((2,0,1))),alpha=0.5)
plt.colorbar()
straincompleto.shape

#### Visualización de el campo de desplazamiento con los vectores de la parte radial del sistema de coordenadas locales

import matplotlib.pyplot as plt
plt.figure(figsize=(100,100))
#plt.quiver(dfi[frame][0,slce,:,:]-dfi[9][0,slce,:,:], dfi[frame][1,slce,:,:]-dfi[9][1,slce,:,:])
plt.quiver(dfi[frame][0,slce,:,:], dfi[frame][1,slce,:,:], scale=600, width=0.001)
plt.quiver(Icoord[frame][2][slce,:,:,0], Icoord[frame][2][slce,:,:,1], color='r',scale=150, width=0.001)
#plt.figure(figsize=(100,100))
#plt.imshow(iec[frame][slce])
