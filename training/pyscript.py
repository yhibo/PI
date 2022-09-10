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
from models.networks import *
import os
import gc
import time

##########################      Normalization     ######################################


def normalize(x, axis=(0, 1, 2)):
    # normalize per volume (x,y,z) frame
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / (sd + 1e-8)

######################### Constants and arguments ######################################

@tf.function
def criterion_netME(y_true, y_pred):
    u = y_pred
    V_0, V_t, M_0, M_t, res = y_true

    V_t_pred = warp(V_t, u)

    M_t_split = tf.split(M_t, M_t.shape[-1], -1)
    M_0_pred  = K.concatenate([warp(K.cast(mt, K.dtype(V_t)), u) for mt in M_t_split], -1)    
    M_0_pred  = keras.activations.softmax(M_0_pred)

    lambda_i = np.array(0.01, dtype= np.float64)
    lambda_a = np.array(0.5, dtype= np.float64)
    lambda_s = np.array(0.1, dtype= np.float64)

    dice = Dice()
    grad = Grad()

    # Intensity loss
    
    L_i = K.mean(K.abs(V_t_pred - V_0))

    # Anatomical loss
    L_a = 0
    L_a += dice.loss(K.cast(M_0==0, dtype=tf.float64), K.cast(M_0_pred==0, dtype=tf.float64))
    L_a += dice.loss(K.cast(M_0==1, dtype=tf.float64), K.cast(M_0_pred==1, dtype=tf.float64))
    L_a += dice.loss(K.cast(M_0==2, dtype=tf.float64), K.cast(M_0_pred==2, dtype=tf.float64))
    L_a += dice.loss(K.cast(M_0==3, dtype=tf.float64), K.cast(M_0_pred==3, dtype=tf.float64))
    L_a = L_a/4.0

    # Smoothness loss
    L_s = grad.loss([],K.cast(u,dtype=tf.float64))

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


    # f = open("training_log.txt", "w")

    # with open("training_log.txt", "a") as f:
        # f.write("GPU used: " + str(tf.config.list_physical_devices('GPU')) + " \n")
        # f.close()


######################### Data loading ######################################
DATA_FOLDER = 'data/training'

volumes_nifti = [nib.load(os.path.join(DATA_FOLDER, f"patient{i:03d}.nii.gz")) for i in range(1, 101)]
segs_nifti = [nib.load(os.path.join(DATA_FOLDER, f"patient{i:03d}_seg.nii.gz")) for i in range(1, 101)]

volumes = [v.get_fdata() for v in volumes_nifti]
segs = [s.get_fdata() for s in segs_nifti]
reslist = [np.array(v.header.get_zooms()[:-1]+(v.shape[-1],), dtype=np.float64) for v in volumes_nifti]

del volumes_nifti, segs_nifti
gc.collect()


V0 = [v[..., 0][None, ..., None] for v in volumes for t in range(1, v.shape[-1])]
Vt = [v[..., t][None, ..., None] for v in volumes for t in range(1, v.shape[-1])]

del volumes
gc.collect()

M0 = [s[..., 0][None, ..., None] for s in segs for t in range(1, s.shape[-1])]
Mt = [s[..., t][None, ..., None] for s in segs for t in range(1, s.shape[-1])]

del segs
gc.collect()

res = [res for res in reslist for t in range(1, int(res[-1]))]

del reslist
gc.collect()


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        loss_value = 0
        for i in range(len(x[0])):
            logits = netME([x[0][i], x[1][i]], training=True)
            loss_value += criterion_netME([y[0][i], y[1][i], y[2][i], y[3][i], y[4][i]], logits)
    grads = tape.gradient(loss_value, netME.trainable_weights)
    netME.optimizer.apply_gradients(zip(grads, netME.trainable_weights))
    return loss_value

@tf.function
def distributed_train_step(x, y):
  per_replica_losses = strategy.run(train_step, args=(x,y))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

######################### Training ######################################

# f.write("Loading data to GPU memory...")

V0_train = tf.data.Dataset.from_tensor_slices(V0)
del V0
gc.collect()

Vt_train = tf.data.Dataset.from_tensor_slices(Vt)
del Vt
gc.collect()

M0_train = tf.data.Dataset.from_tensor_slices(M0)
del M0
gc.collect()

Mt_train = tf.data.Dataset.from_tensor_slices(Mt)
del Mt
gc.collect()

res_train = tf.data.Dataset.from_tensor_slices(res)
del res
gc.collect()


x_train = tf.data.Dataset.zip((V0_train, Vt_train))
y_train = tf.data.Dataset.zip((V0_train, Vt_train, M0_train, Mt_train, res_train))

batch_size = 10
x_train = x_train.shuffle(buffer_size=50).batch(batch_size)
y_train = y_train.shuffle(buffer_size=50).batch(batch_size)
gc.collect()

x_train = strategy.experimental_distribute_dataset(x_train)
y_train = strategy.experimental_distribute_dataset(y_train)
gc.collect()

# f.write("Training started\n")

del V0_train, Vt_train, M0_train, Mt_train, res_train
gc.collect()


for epoch in range(300):
    start_time = time.time()
    print("Start of epoch %d" % (epoch,))
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(zip(x_train, y_train)):
        loss_value = distributed_train_step(x_batch_train, y_batch_train)
        # if step % 10 == 0:
            # f.write("Training loss (for one batch) at step %d: %.4f \n" % (step, float(loss_value)))
            # f.write("Seen so far: %s samples \n" % ((step + 1) * batch_size))
    # f.write("Time taken for epoch %d: %.2fs \n" % (epoch, time.time() - start_time))
    print("Time taken for epoch %d: %.2fs" % (epoch, time.time() - start_time))
# f.write("Saving model\n")


netME.save_weights("netME_weights.h5")
# f.write("Model saved\n")

# f.close()