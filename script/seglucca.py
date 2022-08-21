from tensorflow.keras.models import load_model
from CardIAc_modules.custom_objects import c_o as custom_objects_imported
from CardIAc_modules.models_utils import remove_extra_inputs_outputs
from CardIAc_modules.AISeg_config import logic_IC, label_tissues, myo_density
import os
from natsort import natsorted
import SimpleITK as sitk
import numpy as np
from skimage import transform
from CardIAc_modules.images_utilities import normalize_image, NpArray2VTK
from skimage import morphology
from scipy import ndimage
from datasets.base_dataset import pad_256x256

def get_segmentation(data_folder, patient):

    path_center = 'models/vnet_rlo_2Dcenter.h5'
    path_segment = 'models/GVUNet_cardiac.h5'

    array_images = [
        sitk.ReadImage(os.path.join(data_folder, f))
        for f in natsorted(os.listdir(data_folder))
        if f.endswith(".mhd") and f.startswith(f"{patient}_cSAX_time")
    ]

    # Load models, labels by default: rv(1), myo(2) lv(3)
    model1 = load_model(path_center, custom_objects=custom_objects_imported)
    model2 = load_model(path_segment, custom_objects=custom_objects_imported)

    # Remove extra inputs and extra outputs if they are found
    try:
        model2 = remove_extra_inputs_outputs(model2,logic_IC['cnn_input_label'], logic_IC['cnn_output_label'])
    except:
        pass

    input_shape_m1 = (64,64)
    input_shape_m2 = (128,128)

    frames = 30

    # Center detection (CNN first model)
    frames_shape = (14,256,256) # (z,256,256)
    slice_shape = (256,256) # (256,256)
    volumes = np.zeros((frames,) + frames_shape) # (frames, z, 256, 256)
    volumes = np.array([pad_256x256(sitk.GetArrayFromImage(img).transpose()).transpose() for img in array_images])

    firstFrame = volumes[0] # (z,256,256)
    middleSlice = firstFrame.shape[0]//2
    middleSlice = firstFrame[middleSlice] # (256,256)
    middleSlice = transform.resize(middleSlice, input_shape_m1, mode='symmetric').astype('float32') # (64,64)
    center = normalize_image(middleSlice)
    center = model1.predict(center[np.newaxis,:,:,np.newaxis]).squeeze() # (64,64)

    # Resize to original shape to compute CM
    center = np.round(transform.resize(center, slice_shape, mode='symmetric')).astype(int)

    # Compute centroid of heart (instead of scaling) 
    # (x,y) --> (y,x) swap for slicer is done later
    cm = np.array(ndimage.measurements.center_of_mass(center))
    print("\nCM = ( {}, {} )".format(cm[0], cm[1]))

    spacing = array_images[0].GetSpacing()
    r = int(round(logic_IC['cnn_roi_size']/spacing[0])) # 90 [mm] // 1.25 [pix/mm] = 72 [pix]
    xcm, ycm = cm.round().astype(int)
    xleft, xright = xcm - r, xcm + r
    ydown, yup = ycm - r, ycm + r

    rois = volumes[...,xleft:xright,ydown:yup] # (frames, z, x, y)  
    rois.shape = (rois.shape[0]*rois.shape[1],) + rois.shape[2:] # (frames*z, x, y)

    model2_input = np.zeros((rois.shape[0],) + input_shape_m2) # (frames*z, 128, 128)

    for slice_frame in range(rois.shape[0]):
        I = transform.resize(rois[slice_frame], input_shape_m2, mode='symmetric').astype('float32') # (128,128) 
        model2_input[slice_frame] = normalize_image(I)

    # Second Model: Segmentation
    seg = model2.predict(model2_input[...,np.newaxis]) # (z, 128,128,n_classes)

    # ----------------------------------------------
    # Container for segmentated images
    imgs_seg = np.zeros((rois.shape[0],) + slice_shape, dtype=int) # (frames*z, 256, 256)

    for i in range(seg.shape[0]):    
        # Case of 3 classes

        # This is the order in Net output
        myo_i = seg[i,...,0]
        lv_i = seg[i,...,1]
        rv_i = seg[i,...,2]

        # resize and round (Check this--> Rois shape???? or original shape???)
        rv_i = np.round(transform.resize(rv_i, rois.shape[1:], mode='symmetric')).astype(int)
        myo_i = np.round(transform.resize(myo_i, rois.shape[1:], mode='symmetric')).astype(int)
        lv_i = np.round(transform.resize(lv_i, rois.shape[1:], mode='symmetric')).astype(int)

        # Avoid intersection between rv-myo and lv-myo classes
        rv_myo = rv_i + myo_i
        lv_myo = lv_i + myo_i
        rv_i[rv_myo == 2] = 0
        lv_i[lv_myo == 2] = 0

        # Assign labels (hardcoded TODO: let user select labels) ; 1:RV, 2:myo, 3:LV
        I_seg = label_tissues['RV']*rv_i + label_tissues['myo']*myo_i + label_tissues['LV']*lv_i

        imgs_seg[i,xleft:xright,ydown:yup] = I_seg

    imgs_seg.shape = volumes.shape # (frames, z, 256, 256)

    return imgs_seg.transpose((3,2,1,0))