import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"  # so TF shuts the f up
import SimpleITK as sitk
from tvtk.api import tvtk
import matplotlib.pyplot as plt
import tensorflow as tf
import voxelmorph as vxm
import cv2
import pdb
from natsort import natsorted
from itertools import cycle


def preprocess(fixed, moving, out_shape, slices, feature_tracking):
    # Normalize images to have a N~(0, 1) distribution.
    F, M = sitk.Normalize(fixed), sitk.Normalize(moving)
    if feature_tracking:
        # Get borders
        tmp_f, tmp_m = sitk.GetArrayFromImage(F), sitk.GetArrayFromImage(M)
        (ui_f, uj_f, uk_f), (ui_m, uj_m, uk_m) = np.gradient(tmp_f), np.gradient(tmp_m)
        Fbxy, Mbxy = np.sqrt(uj_f ** 2 + uk_f ** 2), np.sqrt(uj_m ** 2 + uk_m ** 2)
        th_f, th_m = Fbxy.mean(axis=(1, 2)), Mbxy.mean(axis=(1, 2))
        Fbxy[Fbxy < th_f[..., np.newaxis, np.newaxis]] = 0
        Mbxy[Mbxy < th_m[..., np.newaxis, np.newaxis]] = 0
        Fb, Mb = sitk.GetImageFromArray(Fbxy), sitk.GetImageFromArray(Mbxy)
        Fb.CopyInformation(F), Mb.CopyInformation(M)
        F, M = Fb, Mb

    # Match the histograms
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1200)
    matcher.SetNumberOfMatchPoints(20)
    matcher.ThresholdAtMeanIntensityOn()
    M = matcher.Execute(M, F)

    # Resize into out_shape
    f_arr, m_arr = sitk.GetArrayFromImage(F), sitk.GetArrayFromImage(M)
    fixed_preproc, moving_preproc = [], []
    for slice in slices:
        f_array = cv2.resize(f_arr[slice], out_shape, interpolation=cv2.INTER_CUBIC)
        m_array = cv2.resize(m_arr[slice], out_shape, interpolation=cv2.INTER_CUBIC)
        fixed_preproc.append(f_array)
        moving_preproc.append(m_array)
    return fixed_preproc, moving_preproc


def load_data(file_list, out_shape, slices=[5], feature_tracking=False):
    fixed_images, moving_images = [], []
    for f, m in file_list:
        fixed, moving = sitk.ReadImage(f), sitk.ReadImage(m)
        fixed_preprocs, moving_preprocs = preprocess(
            fixed, moving, out_shape, slices, feature_tracking
        )
        for f_preproc, m_preproc in zip(fixed_preprocs, moving_preprocs):
            fixed_images.append(f_preproc), moving_images.append(m_preproc)
    return (
        np.asarray(moving_images)[..., np.newaxis],
        np.asarray(fixed_images)[..., np.newaxis],
    ), (
        np.asarray(fixed_images)[..., np.newaxis],
        np.zeros([len(fixed_images), *out_shape, 2]),
    )


def get_segmentation(folder, patient, out_shape, dilate, slice):
    fname = os.path.join(folder, patient, "AHA", "Healthy", "aha_t0.mhd")
    img = sitk.ReadImage(fname)
    myo = img > 0 
    myo_d = sitk.BinaryDilate(myo, dilate)
    array = sitk.GetArrayFromImage(myo_d)
    return cv2.resize(array[slice], out_shape, interpolation=cv2.INTER_CUBIC)


def get_ed_es_images(folder, patient, out_shape, feature_tracking):
    internal_folder = os.path.join(folder, patient, "SyntheticData", "Healthy")
    fixed = os.path.join(internal_folder, "imageSYNTH_000.vtk")
    moving = os.path.join(internal_folder, "imageSYNTH_011.vtk")
    filenames = [(fixed, moving)]
    return load_data(filenames, out_shape, feature_tracking=feature_tracking)


def get_whole_seq(folder, patient, out_shape, feature_tracking):
    internal_folder = os.path.join(folder, patient, "SyntheticData", "Healthy")
    image_filenames = [
        os.path.join(internal_folder, filename)
        for filename in natsorted(os.listdir(internal_folder))
        if "image" in filename
    ]
    times = [f"00{t}" for t in range(10)] + [f"0{t}" for t in range(10, 30)]
    f_m_filenames = np.asarray(
        [
            (
                filename,
                filename.replace(times[t], times[t + 1] if t != 29 else times[0]),
            )
            for (t, filename) in zip(cycle(range(30)), image_filenames)
        ]
    )
    return load_data(f_m_filenames, out_shape, feature_tracking=feature_tracking)


def get_ed_es_diff_gt(folder, patient, out_shape):
    internal_folder = os.path.join(folder, patient, "SyntheticData", "Healthy")
    ed_img = sitk.ReadImage(os.path.join(internal_folder, "imageSYNTH_000.vtk"))

    reader = tvtk.UnstructuredGridReader()
    reader.file_name = os.path.join(internal_folder, "meshSYNTH_000.vtk")
    reader.update()
    ed_mesh = reader.get_output()

    reader = tvtk.UnstructuredGridReader()
    reader.file_name = os.path.join(internal_folder, "meshSYNTH_011.vtk")
    reader.update()
    es_mesh = reader.get_output()

    disp_img = np.zeros((ed_img.GetWidth(), ed_img.GetHeight(), ed_img.GetDepth(), 3))
    disp_mesh = es_mesh.points.to_array() - ed_mesh.points.to_array()
    for idx, p in enumerate(ed_mesh.points):
        index = ed_img.TransformPhysicalPointToIndex(p)
        if np.any(
            np.asarray([index[1], index[0], index[2]])
            > np.asarray(ed_img.GetSize()) - 1
        ):
            continue
        disp_img[index[1], index[0], index[2]] = disp_mesh[idx] / ed_img.GetSpacing()[0]
    return disp_img


def get_straus_dataset(folder, out_shape, batch_size, feature_tracking):
    internal_folder = "SyntheticData"
    patients = natsorted(
        [os.path.join(folder, f, internal_folder) for f in os.listdir(folder)]
    )
    patients = patients[:9]
    studies = [
        os.path.join(patient, f)
        for patient in patients
        for f in natsorted(os.listdir(patient))
        if os.path.isdir(os.path.join(patient, f))
    ]
    image_filenames = [
        os.path.join(study, filename)
        for study in studies
        for filename in natsorted(os.listdir(study))
        if "image" in filename
    ]
    times = [f"00{t}" for t in range(10)] + [f"0{t}" for t in range(10, 30)]
    f_m_filenames = np.asarray(
        [
            (
                filename,
                filename.replace(times[t], times[t + 1] if t != 29 else times[0]),
            )
            for (t, filename) in zip(cycle(range(30)), image_filenames)
        ]
    )
    idxs = np.arange(len(f_m_filenames))
    np.random.shuffle(idxs)
    train_pairs = cycle(f_m_filenames[idxs])

    def generator(idxs_cycle):
        while True:
            yield load_data(
                [next(idxs_cycle) for _ in range(batch_size)],
                out_shape,
                feature_tracking=feature_tracking,
            )

    return generator(train_pairs)


def get_ed_es_dataset(folder, out_shape, batch_size, feature_tracking):
    internal_folder = "SyntheticData"
    patients = natsorted(
        [os.path.join(folder, f, internal_folder) for f in os.listdir(folder)]
    )
    patients = patients[:9]
    studies = [
        os.path.join(patient, f)
        for patient in patients
        for f in natsorted(os.listdir(patient))
        if os.path.isdir(os.path.join(patient, f))
    ]
    fixed_filenames = [
        os.path.join(study, filename)
        for study in studies
        for filename in natsorted(os.listdir(study))
        if "image" in filename and "000" in filename
    ]
    moving_filenames = [
        os.path.join(study, filename)
        for study in studies
        for filename in natsorted(os.listdir(study))
        if "image" in filename and "011" in filename
    ]
    times = ["000", "011"]
    f_m_filenames = np.asarray(
        [(f, m) for (f, m) in zip(fixed_filenames, moving_filenames)]
    )
    idxs = np.arange(len(f_m_filenames))
    np.random.shuffle(idxs)
    train_pairs = cycle(f_m_filenames[idxs])

    def generator(idxs_cycle):
        while True:
            yield load_data(
                [next(idxs_cycle) for _ in range(batch_size)],
                out_shape,
                feature_tracking=feature_tracking,
            )

    return generator(train_pairs)


def regular_model(features, out_shape, lr, reg_strength=0.01):
    vxm_model = vxm.networks.VxmDense(out_shape, features, int_steps=0)
    losses = ["mse", vxm.losses.Grad("l2").loss]
    loss_weights = [1, reg_strength]
    vxm_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss=losses,
        loss_weights=loss_weights,
    )
    return vxm_model


def diffeomorphic_model(features, out_shape, lr, reg_strength=0.01):
    vxm_model = vxm.networks.VxmDense(out_shape, features, int_downsize=1)
    losses = ["mse", vxm.losses.Grad("l2").loss]
    loss_weights = [1, reg_strength]
    vxm_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss=losses,
        loss_weights=loss_weights,
    )
    return vxm_model


def train(vxm_model, train_generator, epochs, steps_per_epoch, name):
    hist = vxm_model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
    )
    vxm_model.save(name + ".h5")
    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], ".-")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(name + ".pdf")
