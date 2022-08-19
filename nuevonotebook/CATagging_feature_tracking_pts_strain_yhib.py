"""
File: tissue_tracking_strain.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://github.com/curiale
Description:

    Vamos a calcular el strain de los pts del myo a partir de los desplazamientos y el
    modelo aha creado en DatabaseAndModels/CardiacAtlasDataset/CATaggingCineGT2mhd.py
    Estos desplazamientos se calcularon con el codigo  CATagging_feature_tracking.py

    En caso de utilizar la opcion de segmentado se utiliza el modelo aha y sist
    de coord locales que se armo para la segmentación del miocardio en cada t.
    Esto se hace con el archivo ...seg2aha.py
    
"""

import os
import numpy as np
import SimpleITK as sitk
from natsort import natsorted
from config import CA_PHANTOM
from utils.io import cat_read_dfield
from utils.strain_from_motion import cine_dense_strain, temporal_strain_correction
from skimage.morphology import convex_hull_image




import nibabel as nib
from datasets.nifti_dataset import resample_nifti
from scipy.ndimage.measurements import center_of_mass
from utils.utils_aha import *

images_folder = 'segs/our_seg/'
strainellosfolder = 'strain_ellos/our_seg/'

patient = 'v9'
M_nifti = nib.load(os.path.join(images_folder, f"{patient}_seg.nii.gz"))
M_nifti_resampled = resample_nifti(
            M_nifti, order=0, in_plane_resolution_mm=1.25
        )
center = center_of_mass(M_nifti_resampled.get_fdata()[:, :, :, 0] == 1)
M = M_nifti_resampled.get_fdata()
#M = _roll2center_crop(x=M, center=center)
I = np.argmax((M == 1).sum(axis=(0, 1, 3)))
if I > M.shape[2] // 2:
    M = M[:, :, ::-1]
mask = M.transpose((3,2,1,0))
mask = mask==1
mask = mask.astype(float)

myo = sitk.GetImageFromArray(mask[0,:,:,:])
coord_i, data_i = lv_local_coord_system(myo, 0, False)

Icoord_i = sitk.GetArrayFromImage(coord_i)

#  # coord --> (dim, 9): # 3x3 = [c_l,c_c,c_r]
Icoord_i = Icoord_i.reshape(Icoord_i.shape[:3]+(3,3))
ldir = Icoord_i[...,0]
cdir = Icoord_i[...,1]
rdir = Icoord_i[...,2]
cooord = [ldir, cdir, rdir]




demons_std = '_222_221'

method = 'deepstrain'
#method = 'demons'
#method = 'bspline'
use_roi = False
tracking_3d = True
morph_int = False
use_t0 = False # TODO NO ANDA BIEN REVISAR PENSAR !!!
tracking_edges = True # --> Se hizo Feature Tracking. Solo se hizo tracking de bordes

tracking_seg = False
seg_method = 'nn' # metodo usado para seg.

strain_correction = True
smooth_strain = 4
dbtype = 'pts'

if not tracking_seg:
    seg_method = 'ED'
    segsufix = 'segED'



print('Method: {}'.format(method))
print('Use ROI: {}'.format(use_roi))
print('Edges: {}'.format(tracking_edges))
if tracking_seg:
    print('Segmentation: {}'.format(seg_method))
print('Tracking 3D: {}'.format(tracking_3d))
print('Interpolation: {}'.format(morph_int))
print('Strain Correction: {}'.format(strain_correction))


# ------ Patients info
ted = 29 # para todos los pacientes
tes = [10,11,11,11,11,9,9,10,10,9,11,9,10,11,11]

lv_center = np.array([
        [138, 117, 40],
        [138, 117, 40], # Este buscar
        [143, 112, 40],
        [135, 119, 40],
        [118, 125, 40],
        [144, 156, 40],
        [167, 136, 40],
        [108, 148, 40],
        [119, 97, 40],
        [154, 127, 40],
        [110, 145, 40],
        [122, 123, 40],
        [120, 105, 40],
        [93, 132, 40],
        [108, 113, 40],
        ])



# ------ Data config

dataf = os.path.join(CA_PHANTOM, 'MHD_Data')

patients = [f for f in os.listdir(dataf)
        if os.path.isdir(os.path.join(dataf, f)) and f.startswith('v')]
patients = natsorted(patients)

for idp in [7]:#range(len(patients)):
#  for idp in range(0,1):
#  for idp in range(1,2):
#  for idp in range(2,3):
    if tracking_3d:
        msg = '3D'
        strain_sufix = '3d'
    else:
        msg = '2D'
        strain_sufix = '2d'



    sufix = ''
    if morph_int:
        sufix = '-mi'
        msg += ' MI'

    if tracking_edges:
        msg += ' edges'
        strain_sufix += '_edges'



    patient = patients[idp]
    lv_center_i = lv_center[idp,:]



    data = os.path.join(dataf, patient, 'cSAX')

    fnames = [f for f in os.listdir(data) if
            f.startswith('{}_cSAX{}_time_'.format(patient, sufix)) and f.endswith('.mhd')]

    fnames = natsorted(fnames)
    nframes = len(fnames)


    print('Processing patient: {}'.format(patient))

    # ------ Leemos la deformacion y armamos el movimiento
    dfields_folder = 'dfield/our_seg/'
    #dfield = np.load(os.path.join(dfields_folder, f"dfield_demons_256_{patient}.npy"))
    dfield = np.load(os.path.join(dfields_folder, f"dfield_deep_256_{patient}.npy"))

    # ------ Armamos la ROI
    if use_roi:
        roi_s = lv_center_i - 90/dfield[0].GetSpacing()[0]
        roi_e = lv_center_i + 90/dfield[0].GetSpacing()[0]
        roi_s[-1] = 0
        roi_e[-1] = dfield[0].GetSize()[2]

        roi_s = roi_s.round().astype(int)
        roi_e = roi_e.round().astype(int)
        x_s, y_s, z_s = roi_s
        x_e, y_e, z_e = roi_e


    aha = []

    # ------ Leemos el sistema de coordenadas local
    if tracking_seg:

        msg += ' ' + seg_method

        # -------- Leemos bien la seg del miocardio

        # TODO: Temporal esto hay que armar los aha del segmentado y no asi pero lo
        # vamos a hacer asi ahora por simplicidad 
        if seg_method.lower() == 'nn':
            segsufix = 'segNN'
            aha_folder = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'AHA',
                    'SegmentationNN', 'Dilated')
            coord_folder = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Coord',
                    'SegmentationNN', 'Dilated')
        else:
            segsufix = 'segbayes'
            aha_folder = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'AHA',
                    'Segmentation', 'Dilated')
            coord_folder = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'Coord',
                    'Segmentation', 'Dilated')
        myo = []
        Iaha = []
        Icoord = []

        for t in range(1, nframes+1): 
            aha_i = sitk.ReadImage(os.path.join(aha_folder,
                '{}_cSAX_{}_aha_dilated_time_{}.mhd'.format(patient, segsufix, t)))

            coord_i = sitk.ReadImage(os.path.join(coord_folder,
                '{}_cSAX_{}_local_coord_dilated_time_{}.mhd'.format(patient, segsufix, t)))

            myo_i = aha_i > 0

            if use_roi:
                myo_i = myo_i[int(x_s):int(x_e),int(y_s):int(y_e),int(z_s):int(z_e)]
                aha_i = aha_i[int(x_s):int(x_e),int(y_s):int(y_e),int(z_s):int(z_e)]
                coord_i = coord_i[int(x_s):int(x_e),int(y_s):int(y_e),int(z_s):int(z_e)]

            aha.append(aha_i)
            Iaha_i = sitk.GetArrayFromImage(aha_i)
            Icoord_i = sitk.GetArrayFromImage(coord_i)

            #  # coord --> (dim, 9): # 3x3 = [c_l,c_c,c_r]
            Icoord_i = Icoord_i.reshape(Icoord_i.shape[:3]+(3,3))
            ldir = Icoord_i[...,0]
            cdir = Icoord_i[...,1]
            rdir = Icoord_i[...,2]
            Icoord.append([ldir, cdir, rdir])

            myo.append(myo_i)
            Iaha.append(Iaha_i)

    else:
        gt_folder = os.path.join(CA_PHANTOM, 'cMAC', 'GT', 'SSFP', 'GT_Corrected')
        coord_file = os.path.join(gt_folder,
                '{}_corr_local_coord{}_dilated.mhd'.format(patient, sufix))
        coord = sitk.ReadImage(coord_file)

        aha_file = os.path.join(gt_folder,
                '{}_corr_aha{}_dilated.mhd'.format(patient, sufix))
        aha_i = sitk.ReadImage(aha_file)


        # NOTE: Tomaoms la seg original la dilatamos muy poco y armamos el
        # convexhull para luego filtrar aha y coord a los puntos dentro del CH
        myo_file = os.path.join(gt_folder, '{}_corr.mhd'.format(patient))
        myo = sitk.ReadImage(myo_file)
        radii = [2,2,0]
        myo = sitk.BinaryDilate(myo, radii)

        # Armamos el CH
        Ihull = sitk.GetArrayFromImage(myo).astype(np.uint8)

        for ii in range(Ihull.shape[0]): 
            if Ihull[ii].sum() >0:
                Ihull[ii] = convex_hull_image(Ihull[ii])

        chull = sitk.GetImageFromArray(Ihull)
        chull.CopyInformation(aha_i)

        aha_i = aha_i * chull
        myo = aha_i >0

        if use_roi:
            chull = chull[int(x_s):int(x_e),int(y_s):int(y_e),int(z_s):int(z_e)]
            myo = myo[int(x_s):int(x_e),int(y_s):int(y_e),int(z_s):int(z_e)]
            aha_i = aha_i[int(x_s):int(x_e),int(y_s):int(y_e),int(z_s):int(z_e)]
            coord = coord[int(x_s):int(x_e),int(y_s):int(y_e),int(z_s):int(z_e)]


        Icoord = sitk.GetArrayFromImage(coord)
        Ihull = sitk.GetArrayFromImage(chull)
        # coord --> (dim, 9): # 3x3 = [c_l,c_c,c_r]
        Icoord = Icoord.reshape(Icoord.shape[:3]+(3,3))
        ldir = Icoord[...,0] * Ihull[..., np.newaxis]
        cdir = Icoord[...,1] * Ihull[..., np.newaxis]
        rdir = Icoord[...,2] * Ihull[..., np.newaxis]
        Icoord = [ldir, cdir, rdir]

        Imyo = sitk.GetArrayFromImage(myo)
        Iaha = sitk.GetArrayFromImage(aha_i)
        aha.append(aha_i)


    nframes = len(dfield)


    iec_aha = np.zeros((16, nframes+1))
    ier_aha = np.zeros((16, nframes+1))
    ierc_aha = np.zeros((16, nframes+1))


    if tracking_3d:
        iel_aha = np.zeros((16, nframes+1))



    # Ponemos la masa del miocardio que nos da idea de como cambia 
    # rho is  1.05 g/cm^3 = 1.05*1e-3 g/mm^3
    #  rho = np.prod(dfield[0].GetSpacing()) * 1.05 * 1e-3
    #  vol_aha = np.zeros((16, nframes))

    dfaccum = dfield

    iec = []
    ier = []

    for i in range(nframes):
        print('Patient {} Frame {}'.format(patient, i))

        if tracking_seg:
            Iaha_i = Iaha[i]
            Icoord_i = Icoord[i]
            myo_i = myo[i]
        else:
            Iaha_i = Iaha
            Icoord_i = Icoord
            myo_i = myo


        dfield_i = dfield[i]
        roi = myo_i
        Iroi = sitk.GetArrayFromImage(roi) * Iaha_i

        df = dfaccum[i]

        if df.shape[-1] == 2:
            df = [df[...,0], df[...,1]] # [ux, uy]
        elif df.shape[-1] == 3:
            df = [df[...,0], df[...,1], df[...,2]] # [ux, uy, uz]

        ieE_t = cine_dense_strain(df, cooord, mask[0])
        #ieE_t = cine_dense_strain(df, Icoord_i, Iroi)

        # Inf. strain
        iec_t = ieE_t[0]
        ier_t = ieE_t[1]
        ierc_t = ieE_t[2]

        # Eulerian strain
        ec_t = ieE_t[3]
        er_t = ieE_t[4]
        erc_t = ieE_t[5]

        # Lagrangian strain
        Ec_t = ieE_t[6]
        Er_t = ieE_t[7]
        Erc_t = ieE_t[8]

        if tracking_3d:
            iel_t = ieE_t[9]
            el_t = ieE_t[10]
            El_t = ieE_t[11]

        ier.append(ier_t)
        iec.append(iec_t)

        for j in range(16):
            rr,cc,jj = np.where(Iroi == j+1)

            # rho is  1.05 g/cm^3 = 1.05*1e-3 g/mm^3
            #  vol_aha[j, i] = rr.size * rho
            iec_aha[j, i+1] = iec_t[rr,cc,jj].mean() * 100
            ier_aha[j, i+1] = ier_t[rr,cc,jj].mean() * 100
            ierc_aha[j, i+1] = ierc_t[rr,cc,jj].mean() * 100

            if tracking_3d:
                iel_aha[j, i+1] = iel_t[rr,cc,jj].mean() * 100



    if smooth_strain > 0:
        from scipy.ndimage.filters import convolve1d as convolve1d
        iec_aha = convolve1d(iec_aha, np.ones(smooth_strain)/smooth_strain, axis=1)
        ier_aha = convolve1d(ier_aha, np.ones(smooth_strain)/smooth_strain, axis=1)

        # Se colo y lo suavizamos igual 
        #  vol_aha = convolve1d(vol_aha, np.ones(smooth_strain)/smooth_strain, axis=1)
        

        # Ponemos en 0 el t=0 que se modifico levemente al suavizar
        iec_aha[:, 0] = 0
        ier_aha[:, 0] = 0

        if tracking_3d:
            iel_aha = convolve1d(iel_aha, np.ones(smooth_strain)/smooth_strain, axis=1)
            iel_aha[:, 0] = 0



    ie_aha = [iec_aha, ier_aha, ierc_aha]
    if tracking_3d:
        ie_aha.append(iel_aha)

    if strain_correction:
        # Hacemos la correccion y agregamos el 0 por cada region de forma
        # individual
        ie_aha_nofix = ie_aha
        ie_aha = temporal_strain_correction(ie_aha)

        sufix = '_fix' + sufix

    if tracking_3d:
        iec_aha, ier_aha, ierc_aha, iel_aha = ie_aha
    else:
        iec_aha, ier_aha, ierc_aha = ie_aha
    
    # ----------------------------------------------------------------------
    # ----------- Guardamos el strain ---------------------
    # ----------------------------------------------------------------------
    fsave = data.replace('cSAX', 'Strain')
    if not os.path.lexists(fsave):
        os.makedirs(fsave)
   
    strain_sufix += '_ft_' + method + '_' + segsufix + sufix

    if demons_std is not None:
        strain_sufix += demons_std

    fname = os.path.join(fsave, '{}_cSAX_strain_aha_{}.npy'.format(patient, strain_sufix))
    np.save(fname, np.asarray(ie_aha))



    import matplotlib.pyplot as plt
    #  # ----------------------------------------------------------------------------
    #  # -----------  Graficamos el DFied ---------------------
    #  # ----------------------------------------------------------------------------
    # frame=0

    # # -------- Leemos todas las img
    # images = []
    # for t in range(nframes):
    #     img_folder = os.path.join(CA_PHANTOM, 'MHD_Data', patient, 'cSAX')
    #     img = sitk.ReadImage(os.path.join(img_folder, fnames[t]))
    #     if use_roi:
    #         img = img[int(x_s):int(x_e),int(y_s):int(y_e),int(z_s):int(z_e)]
    #     images.append(img)

    # id_slice = 7

    # from utils.util import ScrollingMotionWindows
    # fig, ax = plt.subplots(1, 1)
    # #  tracker = ScroollingMotionWindows(ax, sitk.GetArrayFromImage(images[frame]),
    # #          sitk.GetArrayFromImage(dfield[frame]), mask=Iaha[frame],

    # dfaccumimag = [sitk.GetImageFromArray(dfaccum[i]) for i in range(nframes)]

    # tracker = ScrollingMotionWindows(ax, images, dfaccumimag, masks=aha, frame=frame,
    #         slice_z=id_slice, ch=0, title='{} ({})'.format(patient, seg_method),
    #         cmap_img='gray', cmap='jet', angles='xy',
    #         scale_units='xy',scale=.5)
    # fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    # #  ax.set_title('Time {}'.format(frame))






    # # ----------------------------------------------------------------------------
    # # -----------  Graficamos el Strain---------------------
    # # ----------------------------------------------------------------------------

    # #  xv = np.arange(vol_aha.shape[-1])

    # #  plt.figure()
    # #  plt.plot(xv,vol_aha[:6,:].sum(axis=0), 'b.-',label='Base')
    # #  plt.plot(xv,vol_aha[6:12,:].sum(axis=0), 'r.-',label='Midd.')
    # #  plt.plot(xv,vol_aha[12:,:].sum(axis=0), 'm.-',label='Apex')
    # #  plt.axvline(x=es_p, color='k', linestyle='--')
    # #  plt.title('Myocardial Mass [g] {} {} ({}) (D. {})'.format(method, dbtype, msg, dilate_myo))
    # #  plt.legend()
    # #  plt.grid(True)
    # #  plt.tight_layout()


    # Graficamos el strain
    x = np.arange(ierc_aha.shape[-1])
    es_p = 10
    
    # Solo para nuestro calculo graficamos el strain global radial y circ.

    plt.figure()
    plt.plot(x,ier_aha.mean(axis=0), 'r.-',label='Proposed')
    plt.axvline(x=es_p, color='k', linestyle='--')
    plt.title('{} {} {} Rad. Strain ({})'.format(patient, method, dbtype, msg))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/{}_{}-{}_radStrain_{}.pdf'.format(patient, method,
        dbtype, msg.replace(' ', '_')))


    plt.figure()
    plt.plot(x,iec_aha.mean(axis=0), 'r.-',label='Proposed')
    plt.axvline(x=es_p, color='k', linestyle='--')
    plt.title('{} {} {} Cric. Strain ({})'.format(patient, method, dbtype, msg))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/{}_{}-{}_circStrain_{}.pdf'.format(patient, method,
        dbtype, msg.replace(' ', '_')))


    # if tracking_3d:
    #     plt.figure()
    #     plt.plot(x,iel_aha.mean(axis=0), 'r.-',label='Proposed')
    #     plt.axvline(x=es_p, color='k', linestyle='--')
    #     plt.title('{} {} Long. Strain ({})'.format(method, dbtype, msg))
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig('results/{}_{}-{}_longStrain_{}.pdf'.format(patient, method, dbtype, msg))


#     # Comparamos con INRIA y UPF
#     plt.figure()
#     plt.plot(x,ier_aha.mean(axis=0), 'r.-',label='Proposed')
#     plt.plot(x,ier_aha_inr.mean(axis=0), 'g.-',label='INRIA')
#     plt.plot(x,ier_aha_upf.mean(axis=0), 'b.-',label='UPF')
#     plt.axvline(x=es_p, color='k', linestyle='--')
#     plt.title('{} {} {} Rad. Strain ({})'.format(patient, method, dbtype, msg))
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('results/{}_{}-{}_radStrain_{}_all.pdf'.format(patient, method,
#         dbtype, msg.replace(' ', '_')))


#     plt.figure()
#     plt.plot(x,iec_aha.mean(axis=0), 'r.-',label='Proposed')
#     plt.plot(x,iec_aha_inr.mean(axis=0), 'g.-',label='INRIA')
#     plt.plot(x,iec_aha_upf.mean(axis=0), 'b.-',label='UPF')
#     plt.axvline(x=es_p, color='k', linestyle='--')
#     plt.title('{} {} {} Cric. Strain ({})'.format(patient, method, dbtype, msg))
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('results/{}_{}-{}_circStrain_{}_all.pdf'.format(patient, method,
#         dbtype, msg.replace(' ', '_')))

#     if tracking_3d:
#         plt.figure()
#         plt.plot(x,iel_aha.mean(axis=0), 'r.-',label='Proposed')
#         plt.plot(x,iel_aha_inr.mean(axis=0), 'g.-',label='INRIA')
#         plt.plot(x,iel_aha_upf.mean(axis=0), 'b.-',label='UPF')
#         plt.axvline(x=es_p, color='k', linestyle='--')
#         plt.title('{} {} {} Long. Strain ({})'.format(patient, method, dbtype, msg))
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig('results/{}_{}-{}_longStrain_{}_all.pdf'.format(patient, method,
#             dbtype, msg.replace(' ', '_')))




# # Miramos el radial pero de una region en particular

# id_aha = 0
# plt.figure()
# plt.plot(x,ier_aha[id_aha], 'r.-',label='Proposed')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} {} Rad. Strain ({}) - AHA: {}'.format(patient, method, dbtype,
#     msg, id_aha))
# plt.grid(True)
# plt.tight_layout()




# # Graficamos el promedio de la base med y apice

# plt.figure()
# plt.plot(x,iec_aha[:6,:].mean(axis=0), 'b.-',label='Base')
# plt.plot(x,iec_aha[6:12,:].mean(axis=0), 'r.-',label='Midd.')
# plt.plot(x,iec_aha[12:,:].mean(axis=0), 'm.-',label='Apex')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} {} Cric. Strain ({})'.format(patient, method, dbtype, msg))
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# plt.figure()
# plt.plot(x,ier_aha[:6,:].mean(axis=0), 'b.-',label='Base')
# plt.plot(x,ier_aha[6:12,:].mean(axis=0), 'r.-',label='Midd.')
# plt.plot(x,ier_aha[12:,:].mean(axis=0), 'm.-',label='Apex')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} {} Rad. Strain ({})'.format(patient, method, dbtype, msg))
# plt.legend()
# plt.grid(True)
# plt.tight_layout()





# # Graficamos el promedio de la base med y apice

# plt.figure()
# plt.plot(x,iec_aha_upf[:6,:].mean(axis=0), 'b--',label='Base UPF')
# plt.plot(x,iec_aha_upf[6:12,:].mean(axis=0), 'r--',label='Midd. UPF')
# plt.plot(x,iec_aha_upf[12:,:].mean(axis=0), 'm--',label='Apex UPF')
# plt.plot(x,iec_aha[:6,:].mean(axis=0), 'b.-',label='Base')
# plt.plot(x,iec_aha[6:12,:].mean(axis=0), 'r.-',label='Midd.')
# plt.plot(x,iec_aha[12:,:].mean(axis=0), 'm.-',label='Apex')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} Cric. Strain ({})'.format(method, dbtype, msg))
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# #  plt.savefig('results/{}_{}-{}_BMA-circStrain_{}.pdf'.format(patient, method, dbtype, msg))


# plt.figure()
# plt.plot(x,ier_aha_upf[:6,:].mean(axis=0), 'b--',label='Base UPF')
# plt.plot(x,ier_aha_upf[6:12,:].mean(axis=0), 'r--',label='Midd. UPF')
# plt.plot(x,ier_aha_upf[12:,:].mean(axis=0), 'm--',label='Apex UPF')
# plt.plot(x,ier_aha[:6,:].mean(axis=0), 'b.-',label='Base')
# plt.plot(x,ier_aha[6:12,:].mean(axis=0), 'r.-',label='Midd.')
# plt.plot(x,ier_aha[12:,:].mean(axis=0), 'm.-',label='Apex')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} Rad. Strain ({})'.format(method, dbtype, msg))
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# #  plt.savefig('results/{}_{}-{}_BMA-radStrain_{}.pdf'.format(patient, method, dbtype, msg))




# plt.figure()
# plt.plot(x,iec_aha_upf.mean(axis=0), 'b--',label='UPF')
# plt.plot(x,iec_aha.mean(axis=0), 'r.-',label='Proposed')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} Cric. Strain ({})'.format(method, dbtype, msg))
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('results/{}_{}-{}_circStrain_{}.pdf'.format(patient, method, dbtype, msg))


# plt.figure()
# plt.plot(x,ier_aha_upf.mean(axis=0), 'b--',label='UPF')
# plt.plot(x,ier_aha.mean(axis=0), 'r.-',label='Proposed')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} Rad. Strain ({})'.format(method, dbtype, msg))
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('results/{}_{}-{}_radStrain_{}.pdf'.format(patient, method, dbtype, msg))



# if tracking_3d:
#     plt.figure()
#     plt.plot(x,iel_aha_upf[:6,:].mean(axis=0), 'b--',label='Base UPF')
#     plt.plot(x,iel_aha_upf[6:12,:].mean(axis=0), 'r--',label='Midd. UPF')
#     plt.plot(x,iel_aha_upf[12:,:].mean(axis=0), 'm--',label='Apex UPF')
#     plt.plot(x,iel_aha[:6,:].mean(axis=0), 'b.-',label='Base')
#     plt.plot(x,iel_aha[6:12,:].mean(axis=0), 'r.-',label='Midd.')
#     plt.plot(x,iel_aha[12:,:].mean(axis=0), 'm.-',label='Apex')
#     plt.axvline(x=es_p, color='k', linestyle='--')
#     plt.title('{} {} Long. Strain ({})'.format(method, dbtype, msg))
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('results/{}_{}-{}_BMA-longStrain_{}.pdf'.format(patient, method, dbtype, msg))


# plt.figure()
# plt.plot(x,ierc_aha[:6,:].mean(axis=0), 'b.-',label='Base')
# plt.plot(x,ierc_aha[6:12,:].mean(axis=0), 'r.-',label='Midd.')
# plt.plot(x,ierc_aha[12:,:].mean(axis=0), 'm.-',label='Apex')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} Share Strain (Rad.-Circ) ({})'.format(method, dbtype, msg))
# plt.legend()
# plt.grid(True)




# # Graficamos el promedio del Septum y la pared Lateral
# # Septu aha regions: 2,3,8,9,14
# id_sep = np.array([2,3,8,9,14]) - 1
# # Lateral aha regions: 5,6,11,12,16
# id_lat = np.array([5,6,11,12,16]) - 1

# plt.figure()
# plt.plot(x,iec_aha_upf[id_sep,:].mean(axis=0), 'b--',label='Septum UPF')
# plt.plot(x,iec_aha_upf[id_lat,:].mean(axis=0), 'r--',label='Lateral UPF')
# plt.plot(x,iec_aha[id_sep,:].mean(axis=0), 'b.-',label='Septum')
# plt.plot(x,iec_aha[id_lat,:].mean(axis=0), 'r.-',label='Lateral')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} Cric. Strain ({})'.format(method, dbtype, msg))
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# #  plt.savefig('results/{}_{}-{}_SepLat-circStrain_{}.pdf'.format(patient, method, dbtype, msg))


# plt.figure()
# plt.plot(x,ier_aha_upf[id_sep,:].mean(axis=0), 'b--',label='Septum UPF')
# plt.plot(x,ier_aha_upf[id_lat,:].mean(axis=0), 'r--',label='Lateral UPF')
# plt.plot(x,ier_aha[id_sep,:].mean(axis=0), 'b.-',label='Septum')
# plt.plot(x,ier_aha[id_lat,:].mean(axis=0), 'r.-',label='Lateral')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} Rad. Strain ({})'.format(method, dbtype, msg))
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# #  plt.savefig('results/{}_{}-{}_SepLat-radStrain_{}.pdf'.format(patient, method, dbtype, msg))


# if tracking_3d:
#     plt.figure()
#     plt.plot(x,iel_aha_upf[id_sep,:].mean(axis=0), 'b--',label='Septum UPF')
#     plt.plot(x,iel_aha_upf[id_lat,:].mean(axis=0), 'r--',label='Lateral UPF')
#     plt.plot(x,iel_aha[id_sep,:].mean(axis=0), 'b.-',label='Septum')
#     plt.plot(x,iel_aha[id_lat,:].mean(axis=0), 'r.-',label='Lateral')
#     plt.axvline(x=es_p, color='k', linestyle='--')
#     plt.title('{} {} Long. Strain ({})'.format(method, dbtype, msg))
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('results/{}_{}-{}_SepLat-longStrain_{}.pdf'.format(patient, method, dbtype, msg))

# plt.figure()
# plt.plot(x,ierc_aha[id_sep,:].mean(axis=0), 'b.-',label='Septum')
# plt.plot(x,ierc_aha[id_lat,:].mean(axis=0), 'r.-',label='Lateral')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} Share Strain (Rad.-Circ) ({})'.format(method, dbtype, msg))
# plt.legend()
# plt.grid(True)



# # Graficamos dos segmentos opuestos 8 vs 11
# id8 = 8
# id11 = 11


# plt.figure()
# plt.plot(x,iec_aha[id8 -1,:], 'b.-',label='Segment 8')
# plt.plot(x,iec_aha[id11 -1,:], 'r.-',label='Segment 11')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} Cric. Strain ({})'.format(method, dbtype, msg))
# plt.legend()
# plt.grid(True)


# plt.figure()
# plt.plot(x,ier_aha[id8 -1,:], 'b.-',label='Segment 8')
# plt.plot(x,ier_aha[id11 -1,:], 'r.-',label='Segment 11')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} Rad. Strain ({})'.format(method, dbtype, msg))
# plt.legend()
# plt.grid(True)


# plt.figure()
# plt.plot(x,iel_aha[id8 -1,:], 'b.-',label='Segment 8')
# plt.plot(x,iel_aha[id11 -1,:], 'r.-',label='Segment 11')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} Long. Strain ({})'.format(method, dbtype, msg))
# plt.legend()
# plt.grid(True)

# plt.figure()
# plt.plot(x,ierc_aha[id8 -1,:], 'b.-',label='Segment 8')
# plt.plot(x,ierc_aha[id11 -1,:], 'r.-',label='Segment 11')
# plt.axvline(x=es_p, color='k', linestyle='--')
# plt.title('{} {} Share Strain (Rad.-Circ) ({})'.format(method, dbtype, msg))
# plt.legend()
# plt.grid(True)



# label = 1

# iecm = np.array([iec[i][mask[0]==label].mean() for i in range(nframes)]) * 100
# ierm = np.array([ier[i][mask[0]==label].mean() for i in range(nframes)]) * 100

# plt.figure()
# plt.title('SEG PAPER Y CAMPO DEEP')
# strainellos = np.load(os.path.join(strainellosfolder, f"strain_{patient}.npy"))
# plt.plot(iecm-iecm[-1]*np.arange(nframes)/(nframes-1), color='c', label="C-nuestro", marker='.')
# plt.legend()
# plt.figure()
# plt.title('SEG PAPER Y CAMPO DEEP')
# plt.plot(ierm-ierm[-1]*np.arange(nframes)/(nframes-1), color='r', label="R-nuestro", marker='.')
# #plt.plot(strainellos[:,1], color='c', label="C-ellos", )
# #plt.plot(strainellos[:,0], color='r', label="R-ellos")
# plt.legend()


plt.figure(figsize=(10,10))
#plt.quiver(montage(dfi[10][0,:,:,:]), montage(dfi[10][1,:,:,:]), scale=100000000)
plt.imshow(np.array(Iroi[6,:,:]>0)*2 + np.array(mask[0][6,:,:]>0))
plt.quiver(dfaccum[10][6,:,:,0], dfaccum[10][6,:,:,1], scale=1000, angles='xy')


plt.show()
