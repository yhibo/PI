def get_segmentation(cine):
    try:
        os.mkdir(os.path.join(out_folder, "images"))
    except OSError:
        pass

    opt = CarSON_options()
    model = deep_strain_model.DeepStrain(Adam, opt=opt)
    netS = model.get_netS()

    V = cine.get_fdata()
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