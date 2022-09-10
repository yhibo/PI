from models import deep_strain_model
from datasets import base_dataset
from datasets.nifti_dataset import resample_nifti
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import center_of_mass
from datasets.base_dataset import roll_and_pad_256x256_to_center_inv
from datasets.base_dataset import _roll2center_crop
from scipy.ndimage import gaussian_filter
import numpy as np
import SimpleITK as sitk
from utils.strain_from_motion import *
from utils.utils_aha import *
import nibabel as nib
from skimage.util import montage
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K 
from models.dense_image_warp import dense_image_warp3d as warp
from models.networks_fit import *
import os
import gc
import time

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

##########################      Normalization     ######################################


def normalize(x, axis=(0, 1, 2)):
    # normalize per volume (x,y,z) frame
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / (sd + 1e-8)

######################### Constants and arguments ######################################

@tf.function
def criterion_netME(y_true, y_pred):

    print("y_true shape: ", tf.shape(y_true))
    print("y_pred shape: ", tf.shape(y_pred))

    u = y_pred
    V_0 = tf.expand_dims(y_true[:,0], axis=-1)
    V_t = tf.expand_dims(y_true[:,1], axis=-1)
    M_0 = tf.expand_dims(y_true[:,2], axis=-1)
    M_t = tf.expand_dims(y_true[:,3], axis=-1)
    res = tf.expand_dims(y_true[:,4], axis=-1)

    V_0_pred = warp(V_t, u)

    M_t_split = tf.split(M_t, M_t.shape[-1], -1)
    M_0_pred  = K.concatenate([warp(K.cast(mt, K.dtype(V_t)), u) for mt in M_t_split], -1)    
    M_0_pred  = keras.activations.softmax(M_0_pred)

    lambda_i = np.array(0.01, dtype= np.float32)
    lambda_a = np.array(0.5, dtype= np.float32)
    lambda_s = np.array(0.1, dtype= np.float32)

    dice = Dice()
    grad = Grad()

    # Intensity loss
    
    L_i = K.mean(K.abs(V_0_pred - V_0), axis=(1,2,3,4))

    # Anatomical loss
    L_a = 0
    L_a += dice.loss(K.cast(M_0==0, dtype=tf.float32), K.cast(M_0_pred==0, dtype=tf.float32))
    L_a += dice.loss(K.cast(M_0==1, dtype=tf.float32), K.cast(M_0_pred==1, dtype=tf.float32))
    L_a += dice.loss(K.cast(M_0==2, dtype=tf.float32), K.cast(M_0_pred==2, dtype=tf.float32))
    L_a += dice.loss(K.cast(M_0==3, dtype=tf.float32), K.cast(M_0_pred==3, dtype=tf.float32))
    L_a = L_a/4.0

    # Smoothness loss
    resux = tf.ones(tf.shape(u)[:-1], dtype=tf.float32)
    resuy = tf.ones(tf.shape(u)[:-1], dtype=tf.float32)
    resuz = tf.ones(tf.shape(u)[:-1], dtype=tf.float32)
    resu = tf.stack([resux, resuy, resuz], axis=-1)
    resu = u*resu
    L_s = grad.loss([],K.cast(resu,dtype=tf.float32))

    return lambda_i * L_i + lambda_a * L_a + lambda_s * L_s

class CarMEN_options:
    def __init__(self):
        self.isTrain = True
        self.volume_shape = (128, 128, 16, 1)
        self.criterion_netME = criterion_netME
        self.netME_lr = 1e-4


tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
opt = CarMEN_options()

with strategy.scope():
    model = deep_strain_model.DeepStrain(Adam, opt=opt)
    netME = model.get_netME()




######################### Data loading ######################################
DATA_FOLDER = 'data/training'

volumes_nifti = [nib.load(os.path.join(DATA_FOLDER, f"patient{i:03d}.nii.gz")) for i in range(1, 101)]
segs_nifti = [nib.load(os.path.join(DATA_FOLDER, f"patient{i:03d}_seg.nii.gz")) for i in range(1, 101)]

volumes = [v.get_fdata() for v in volumes_nifti]
segs = [s.get_fdata() for s in segs_nifti]
reslist = [np.array(v.header.get_zooms()[:-1]+(v.shape[-1],), dtype=np.float32).reshape(4,1,1,1) for v in volumes_nifti]

del volumes_nifti, segs_nifti
gc.collect()


V0 = np.array([v[..., 0][..., np.newaxis] for v in volumes for t in range(1, v.shape[-1])])
Vt = np.array([v[..., t][..., np.newaxis] for v in volumes for t in range(1, v.shape[-1])])

del volumes
gc.collect()

M0 = np.array([s[..., 0][..., np.newaxis] for s in segs for t in range(1, s.shape[-1])])
Mt = np.array([s[..., t][..., np.newaxis] for s in segs for t in range(1, s.shape[-1])])

del segs
gc.collect()

print(reslist[0].shape)
print(int(reslist[0][-1].squeeze()))
res = np.array([np.pad(res[..., 0], ((0,124),(0,127),(0,15)), constant_values=(1,))[..., np.newaxis] for res in reslist for t in range(1, int(res[-1].squeeze()))])
print(res[0].shape)

del reslist
gc.collect()

x = [V0, Vt]
y = np.array([np.array((v0, vt, m0, mt, r)) for v0,vt,m0,mt,r in zip(V0, Vt, M0, Mt, res)])

netME.fit(x, y, epochs=100, batch_size=5, verbose=1)

netME.save_weights("netME_weights_fit.h5")
