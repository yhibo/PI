

def get_motion(cine, seg, results_folder):

    try:
        os.mkdir(results_folder)
    except OSError:
        pass
    opt = CarMEN_options()
    model = deep_strain_model.DeepStrain(Adam, opt=opt)
    netME = model.get_netME()

    # logging.info(f"Motion on patient {patient}.")
    # V_nifti = nib.load(os.path.join(images_folder, f"{patient}.nii.gz"))
    # M_nifti = nib.load(os.path.join(images_folder, f"{patient}_seg.nii.gz"))

    # V_nifti_resampled = resample_nifti(
    #     V_nifti, order=1, in_plane_resolution_mm=1.25, number_of_slices=16
    # )
    # M_nifti_resampled = resample_nifti(
    #     M_nifti, order=0, in_plane_resolution_mm=1.25, number_of_slices=16
    # )

    # center = center_of_mass(M_nifti_resampled.get_fdata()[:, :, :, 0] == 1)
    # V = V_nifti_resampled.get_fdata()
    # M = M_nifti_resampled.get_fdata()

    # V = _roll2center_crop(x=V, center=center)
    # M = _roll2center_crop(x=M, center=center)

    # I = np.argmax((M == 1).sum(axis=(0, 1, 3)))
    # if I > M.shape[2] // 2:
    #     print("flip")
    #     V = V[:, :, ::-1]
    #     M = M[:, :, ::-1]

    V = cine.get_fdata()
    V = normalize(V, axis=(0, 1, 2))
    mask_end_diastole = seg[..., 0]
    y_t = []

    for t in range(30):
        V_0 = V[..., 0][None, ..., None]
        V_t = V[..., t][None, ..., None]
        y_t.append(gaussian_filter(netME([V_0, V_t]).numpy(), sigma=(0,2,2,0,0)))
        strain = myocardial_strain.MyocardialStrain(
            mask=mask_end_diastole, flow=y_t[t][0, :, :, :, :]
        )

    np.save(os.path.join(results_folder, f"dfield_{patient}.npy"), np.asarray(y_t))

    return np.asarray(y_t)