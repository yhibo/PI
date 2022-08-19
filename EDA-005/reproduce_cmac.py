################################## Imports #############################################
import argparse
import logging
import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import vtk
from natsort import natsorted
from scipy.ndimage import gaussian_filter
from scipy.ndimage.measurements import center_of_mass
from tensorflow.keras.optimizers import Adam
from vtk.numpy_interface import dataset_adapter as dsa

from datasets import base_dataset
from datasets.base_dataset import roll_and_pad_256x256_to_center_inv
from datasets.base_dataset import _roll2center_crop
from datasets.nifti_dataset import resample_nifti
from models import deep_strain_model
from utils import myocardial_strain

from scipy.ndimage import interpolation


######################### Constants and arguments ######################################


logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")

DATASET_FOLDER = "/Volumes/Untitled/PI/CMAC/MHD_Data"
LMKS_FOLDER = "/Volumes/Untitled/PI/CMAC/cMAC/GT/SSFP"
patients = [f"v{p}" for p in range(1, 17) if p != 3]
es_times = [10, 11, 11, 11, 11, 9, 9, 10, 10, 9, 11, 9, 10, 11, 11]
ES = {k: v for (k, v) in zip(patients, es_times)}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--convert_to_nifti", type=bool, default=True)
    parser.add_argument("--use_our_segs", type=bool, default=False)
    parser.add_argument("--data_folder", type=str, default=DATASET_FOLDER)
    parser.add_argument("--out_folder", type=str, default=".")
    return parser.parse_args()


class CarSON_options:
    def __init__(self):
        self.isTrain = False
        self.image_shape = (128, 128, 1)
        self.nlabels = 4
        self.pretrained_models_netS = "models/carson_Jan2021.h5"
        self.pretrained_models_netME = "models/carmen_Jan2021.h5"


class CarMEN_options:
    def __init__(self):
        self.isTrain = False
        self.volume_shape = (128, 128, 16, 1)
        self.pretrained_models_netS = "models/carson_Jan2021.h5"
        self.pretrained_models_netME = "models/carmen_Jan2021.h5"


################################# Main loop ############################################


def main():
    args = parse_arguments()
    get_images(args.data_folder, args.out_folder, args.use_our_segs)
    if not args.use_our_segs:
        get_segmentation(args.out_folder)
    # TODO: Save as bsplines through configuration
    get_strain_and_motion(args.out_folder)
    print(get_strain_values(args.out_folder))
    print(get_motion_error(args.out_folder))


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


############################## MHD to Nifti ############################################


def convert_mhd_to_nifti(data_folder, out_file, patient):
    images = [
        sitk.ReadImage(os.path.join(data_folder, f))
        for f in natsorted(os.listdir(data_folder))
        if f.endswith(".mhd") and f.startswith(f"{patient}_cSAX_time")
    ]
    sitk.WriteImage(sitk.JoinSeries(images), out_file)



def get_images(data_folder, out_folder, use_our_segs):
    try:
        os.mkdir(os.path.join(out_folder, "images"))
    except OSError:
        pass
    for patient in patients:
        logging.info(f"Images to nifti on patient {patient}")
        cmac_folder = os.path.join(data_folder, patient, "cSAX")
        out_file = os.path.join(out_folder, "images", f"{patient}.nii.gz")
        convert_mhd_to_nifti(cmac_folder, out_file, patient)
        if use_our_segs:
            seg_folder = os.path.join(data_folder, patient, "SegmentationNN")
            out_file = os.path.join(out_folder, "images", f"{patient}_seg.nii.gz")
            convert_mhd_to_nifti(seg_folder, out_file, patient)


def get_images_2(data_folder, out_folder, use_our_segs):
    try:
        os.mkdir(os.path.join(out_folder, "images"))
    except OSError:
        pass
    for patient in patients:
        logging.info(f"Images to nifti on patient {patient}")
        cmac_folder = os.path.join(data_folder, patient, "cSAX")
        out_file = os.path.join(out_folder, "images", f"{patient}.nii.gz")
        csax_images = (
            [
                sitk.ReadImage(os.path.join(cmac_folder, f))
                for f in natsorted(os.listdir(cmac_folder))
                if f.endswith(".mhd") and f.startswith(f"{patient}_cSAX_time")
            ]
        )
        origin = csax_images[0].GetOrigin()
        spacing = csax_images[0].GetSpacing()

        sitk.WriteImage(sitk.JoinSeries(csax_images), out_file)

        if use_our_segs:
            seg_folder = os.path.join(data_folder, patient, "SegmentationNN")
            out_file = os.path.join(out_folder, "images", f"{patient}_seg.nii.gz")
            seg_images = (
                [
                    sitk.ReadImage(os.path.join(seg_folder, f))
                    for f in natsorted(os.listdir(seg_folder))
                    if f.endswith(".mhd") and f.startswith(f"{patient}_cSAX")
                ]
            )
            for seg_im in seg_images:
                seg_im.SetOrigin(origin)
                seg_im.SetSpacing(spacing)
            sitk.WriteImage(sitk.JoinSeries(seg_images), out_file)


##############################  Segmentation ###########################################


def get_segmentation(out_folder):
    try:
        os.mkdir(os.path.join(out_folder, "images"))
    except OSError:
        pass

    opt = CarSON_options()
    model = deep_strain_model.DeepStrain(Adam, opt=opt)
    netS = model.get_netS()

    for patient in patients:
        logging.info(f"Segmentation on patient {patient}")
        V_nifti = nib.load(os.path.join(out_folder, "images", f"{patient}.nii.gz"))
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
        M_nifti.to_filename(os.path.join(out_folder, "images", f"{patient}_seg.nii.gz"))


############################## Strain and motion #######################################


def get_strain_and_motion(out_folder):
    results_folder = os.path.join(out_folder, "results")
    images_folder = os.path.join(out_folder, "images")
    try:
        os.mkdir(results_folder)
    except OSError:
        pass
    opt = CarMEN_options()
    model = deep_strain_model.DeepStrain(Adam, opt=opt)
    netME = model.get_netME()

    for patient in patients:
        logging.info(f"Motion and strain on patient {patient}.")
        V_nifti = nib.load(os.path.join(images_folder, f"{patient}.nii.gz"))
        M_nifti = nib.load(os.path.join(images_folder, f"{patient}_seg.nii.gz"))

        V_nifti_resampled = resample_nifti(
            V_nifti, order=1, in_plane_resolution_mm=1.25, number_of_slices=16
        )
        M_nifti_resampled = resample_nifti(
            M_nifti, order=0, in_plane_resolution_mm=1.25, number_of_slices=16
        )

        center = center_of_mass(M_nifti_resampled.get_fdata()[:, :, :, 0] == 2)
        V = V_nifti_resampled.get_fdata()
        M = M_nifti_resampled.get_fdata()

        V = _roll2center_crop(x=V, center=center)
        M = _roll2center_crop(x=M, center=center)

        I = np.argmax((M == 1).sum(axis=(0, 1, 3)))
        if I > M.shape[2] // 2:
            V = V[:, :, ::-1]
            M = M[:, :, ::-1]

        V = normalize(V, axis=(0, 1, 2))
        mask_end_diastole = M[..., 0]
        y_t = []
        strain_t = []
        straincompleto_t = []
        mask_rot_t = []
        for t in range(30):
            V_0 = V[..., 0][None, ..., None]
            V_t = V[..., t][None, ..., None]
            y_t.append(gaussian_filter(netME([V_0, V_t]).numpy(), sigma=(0,2,2,0,0)))
            strain = myocardial_strain.MyocardialStrain(
                mask=mask_end_diastole, flow=y_t[t][0, :, :, :, :]
            )
            strain.calculate_strain(lv_label=2)

            # strain_t.append(
            #     [
            #         strain.Err,
            #         strain.Ecc,
            #         strain.Erc,
            #         strain.Ecr
            #     ]
            # )

            straincompleto_t.append(
                [
                    strain.Err,
                    strain.Ecc,
                ]
            )

            strain_t.append(
                [
                    strain.Err[strain.mask_rot == 2].mean(),
                    strain.Ecc[strain.mask_rot == 2].mean(),
                ]
            )

            mask_rot_t.append(strain.mask_rot)

        np.save(os.path.join(results_folder, f"dfield_{patient}.npy"), np.asarray(y_t))
        np.save(os.path.join(results_folder, f"strain_{patient}.npy"), np.asarray(strain_t))
        np.save(os.path.join(results_folder, f"completo_strain_{patient}.npy"), np.asarray(straincompleto_t))
        np.save(os.path.join(results_folder, f"mask_rot_{patient}.npy"), np.asarray(mask_rot_t))


def get_strain_and_motion_2(out_folder):
    results_folder = os.path.join(out_folder, "results")
    images_folder = os.path.join(out_folder, "images")
    try:
        os.mkdir(results_folder)
    except OSError:
        pass
    opt = CarMEN_options()
    model = deep_strain_model.DeepStrain(Adam, opt=opt)
    netME = model.get_netME()

    for patient in patients:
        logging.info(f"Motion and strain on patient {patient}.")
        V_nifti = nib.load(os.path.join(images_folder, f"{patient}.nii.gz"))
        M_nifti = nib.load(os.path.join(images_folder, f"{patient}_seg.nii.gz"))

        V_nifti_resampled = resample_nifti(
            V_nifti, order=1, in_plane_resolution_mm=1.25, number_of_slices=16
        )
        M_nifti_resampled = resample_nifti(
            M_nifti, order=0, in_plane_resolution_mm=1.25, number_of_slices=16
        )

        center = center_of_mass(M_nifti_resampled.get_fdata()[:, :, :, 0] == 1)
        V = V_nifti_resampled.get_fdata()
        M = M_nifti_resampled.get_fdata()

        V = _roll2center_crop(x=V, center=center)
        M = _roll2center_crop(x=M, center=center)

        I = np.argmax((M == 1).sum(axis=(0, 1, 3)))
        if I > M.shape[2] // 2:
            print("flip")
            V = V[:, :, ::-1]
            M = M[:, :, ::-1]

        V = normalize(V, axis=(0, 1, 2))
        mask_end_diastole = M[..., 0]
        y_t = []
        strain_t = []
        straincompleto_t = []
        mask_rot_t = []
        for t in range(30):
            V_0 = V[..., 0][None, ..., None]
            V_t = V[..., t][None, ..., None]
            y_t.append(gaussian_filter(netME([V_0, V_t]).numpy(), sigma=(0,2,2,0,0)))
            strain = myocardial_strain.MyocardialStrain(
                mask=mask_end_diastole, flow=y_t[t][0, :, :, :, :]
            )
            strain.calculate_strain(lv_label=1)

            # strain_t.append(
            #     [
            #         strain.Err,
            #         strain.Ecc,
            #         strain.Erc,
            #         strain.Ecr
            #     ]
            # )

            straincompleto_t.append(
                [
                    strain.Err,
                    strain.Ecc,
                ]
            )

            strain_t.append(
                [
                    strain.Err[strain.mask_rot == 1].mean(),
                    strain.Ecc[strain.mask_rot == 1].mean(),
                ]
            )

            mask_rot_t.append(strain.mask_rot)

        np.save(os.path.join(results_folder, f"dfield_{patient}.npy"), np.asarray(y_t))
        np.save(os.path.join(results_folder, f"strain_{patient}.npy"), np.asarray(strain_t))
        np.save(os.path.join(results_folder, f"completo_strain_{patient}.npy"), np.asarray(straincompleto_t))
        np.save(os.path.join(results_folder, f"mask_rot_{patient}.npy"), np.asarray(mask_rot_t))


def get_strain_and_motion_3(out_folder):
    results_folder = os.path.join(out_folder, "results")
    images_folder = os.path.join(out_folder, "images")
    try:
        os.mkdir(results_folder)
    except OSError:
        pass
    opt = CarMEN_options()
    model = deep_strain_model.DeepStrain(Adam, opt=opt)
    netME = model.get_netME()

    patient = 'v9'

    logging.info(f"Motion and strain on patient {patient}.")
    V_nifti = nib.load(os.path.join(images_folder, f"{patient}.nii.gz"))
    M_nifti = nib.load(os.path.join(images_folder, f"{patient}_seg.nii.gz"))

    V_nifti_resampled = resample_nifti(
        V_nifti, order=1, in_plane_resolution_mm=1.25, number_of_slices=16
    )
    M_nifti_resampled = resample_nifti(
        M_nifti, order=0, in_plane_resolution_mm=1.25, number_of_slices=16
    )

    center = center_of_mass(M_nifti_resampled.get_fdata()[:, :, :, 0] == 1)
    V = V_nifti_resampled.get_fdata()
    M = M_nifti_resampled.get_fdata()

    V = _roll2center_crop(x=V, center=center)
    M = _roll2center_crop(x=M, center=center)

    I = np.argmax((M == 1).sum(axis=(0, 1, 3)))
    if I > M.shape[2] // 2:
        print("flip")
        V = V[:, :, ::-1]
        M = M[:, :, ::-1]

    V = normalize(V, axis=(0, 1, 2))
    mask_end_diastole = M[..., 0]
    y_t = []
    strain_t = []
    straincompleto_t = []
    mask_rot_t = []
    for t in range(30):
        V_0 = V[..., 0][None, ..., None]
        V_t = V[..., t][None, ..., None]
        df = gaussian_filter(netME([V_0, V_t]).numpy(), sigma=(0,2,2,0,0))[0]
        strain = myocardial_strain.MyocardialStrain(
            mask=mask_end_diastole, flow=df[:, :, :, :]
        )
        df = interpolation.zoom(df, [1, 1, 14.0/16.0, 1], order=1)
        nifti_info = {'center_resampled' : center,
                      'center_resampled_256x256' : (128,128),
                      'shape_resampled' : (256,256,14)}
        y_t.append(roll_and_pad_256x256_to_center_inv(df, nifti_info))

        strain.calculate_strain(lv_label=1)

        straincompleto_t.append(
            [
                strain.Err,
                strain.Ecc,
            ]
        )

        strain_t.append(
            [
                strain.Err[strain.mask_rot == 1].mean(),
                strain.Ecc[strain.mask_rot == 1].mean(),
            ]
        )

        mask_rot_t.append(strain.mask_rot)

    np.save(os.path.join(results_folder, f"dfield_256_{patient}.npy"), np.asarray(y_t))
    # np.save(os.path.join(results_folder, f"strain_{patient}.npy"), np.asarray(strain_t))
    # np.save(os.path.join(results_folder, f"completo_strain_{patient}.npy"), np.asarray(straincompleto_t))
    # np.save(os.path.join(results_folder, f"mask_rot_{patient}.npy"), np.asarray(mask_rot_t))



############################### Validations ############################################


def get_strain_values(out_folder):
    strain_folder = os.path.join(out_folder, "results")
    str_results = {"Err": 0, "Ecc": 0, "Srr_d": 0, "Srr_s": 0, "Scc_d": 0, "Scc_s": 0}
    for patient in patients:
        strain = np.load(os.path.join(strain_folder, f"strain_{patient}.npy"))
        strain_rate = np.gradient(strain, axis=0)
        str_results["Err"] += 100 * strain[ES[patient], 0] / len(patients)
        str_results["Ecc"] += 100 * strain[ES[patient], 1] / len(patients)
        str_results["Srr_d"] += np.max(strain_rate[..., 0]) / len(patients) * 30
        str_results["Scc_d"] += np.min(strain_rate[..., 1]) / len(patients) * 30
        str_results["Srr_s"] += np.min(strain_rate[..., 0]) / len(patients) * 30
        str_results["Scc_s"] += np.max(strain_rate[..., 1]) / len(patients) * 30
    return str_results


def get_motion_error(out_folder):
    images_folder = os.path.join(out_folder, "images")
    dfield_folder = os.path.join(out_folder, "results")
    error = []
    for patient in patients:
        # Load landmarks
        lmks_folder = f"{LMKS_FOLDER}/{patient}/LMKS_fix/VTK_COORDINATES"
        points = {}
        for t in (0, ES[patient]):
            reader = vtk.vtkGenericDataObjectReader()
            reader.SetFileName(os.path.join(lmks_folder, f"obs1_groundTruth{t:03}.vtk"))
            reader.Update()
            polydata = reader.GetOutput()
            points[t] = dsa.WrapDataObject(polydata).Points
        
        # Load images and dfield
        img = sitk.ReadImage(os.path.join(images_folder, f"{patient}.nii.gz"))
        M_nifti = nib.load(os.path.join(images_folder, f"{patient}_seg.nii.gz"))
        dfield = np.load(os.path.join(dfield_folder, f"dfield_{patient}.npy"))
        
        # Get new origin
        M_nifti_resampled = resample_nifti(
            M_nifti, order=0, in_plane_resolution_mm=1.25, number_of_slices=16
        )
        center = center_of_mass(M_nifti_resampled.get_fdata()[:, :, :, 0] == 2)
        spacing = np.asarray(img.GetSpacing()[:-1])
        origin = np.asarray(img.GetOrigin()[:-1])
        new_origin = np.round(center - np.asarray([64, 64, 8])) * spacing + origin

        # For each point in 
        for ed_point, es_point in zip(points[0], points[ES[patient]]):
            index = ((ed_point - new_origin) / spacing).astype(int)
            df = dfield[ES[patient], 0, index[0], index[1], index[2], :]
            moved_point = ed_point + df * spacing
            error.append(np.linalg.norm(moved_point[:-1]-es_point[:-1]))
    return error
