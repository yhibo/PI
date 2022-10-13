import os
from re import M
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from models.dense_image_warp import dense_image_warp3d as warp
from models.networks import *
from models import deep_strain_model
from tensorflow.keras.optimizers import Adam
import time

#tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')

# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

strategy = tf.distribute.MirroredStrategy(gpus)


##########################      Model Options     ######################################

@tf.function
def criterion_netME(y_true, y_pred):

    u = y_pred
    V_0 = tf.expand_dims(y_true[:,0], axis=-1)
    V_t = tf.expand_dims(y_true[:,1], axis=-1)
    M_0 = tf.expand_dims(y_true[:,2], axis=-1)
    M_t = tf.expand_dims(y_true[:,3], axis=-1)
    resx = tf.expand_dims(y_true[:,4], axis=-1)
    resy = tf.expand_dims(y_true[:,5], axis=-1)
    resz = tf.expand_dims(y_true[:,6], axis=-1)

    V_0_pred = warp(V_t, u)

    # M_t_split = tf.split(M_t, M_t.shape[-1], -1)
    # M_0_pred  = K.concatenate([warp(K.cast(mt, K.dtype(V_t)), u) for mt in M_t_split], -1)
    M_0_pred = warp(M_t, u)
    M_0_pred = tf.round(M_0_pred)
    # M_0_pred  = keras.activations.softmax(M_0_pred)

    lambda_i = np.array(0.001, dtype= np.float32)
    lambda_a = np.array(0.05, dtype= np.float32)
    lambda_s = np.array(0.01, dtype= np.float32)

    dice = Dice()
    grad = Grad()

    # Intensity loss
    
    L_i = K.mean((V_0_pred - V_0)**2, axis=(1,2,3,4))

    # Anatomical loss
    L_a = dice.loss(K.cast(M_0==0, 'float32'), K.cast(M_0_pred==0, 'float32'))
    L_a += dice.loss(K.cast(M_0==1, 'float32'), K.cast(M_0_pred==1, 'float32'))
    L_a += dice.loss(K.cast(M_0==2, 'float32'), K.cast(M_0_pred==2, 'float32'))
    L_a += dice.loss(K.cast(M_0==3, 'float32'), K.cast(M_0_pred==3, 'float32'))
    L_a /= 4
    # L_a = K.mean((M_0_pred - M_0)**2, axis=(1,2,3,4))

    # Smoothness loss
    res = tf.concat([resx, resy, resz], axis=-1)
    L_s = grad.loss([],u*res)

    return lambda_i * L_i + lambda_a * L_a + lambda_s * L_s + lambda_a

class CarMEN_options:
    def __init__(self):
        self.isTrain = True
        self.volume_shape = (128, 128, 16, 1)
        self.criterion_netME = criterion_netME
        self.netME_lr = 1e-4


##########################      DATA     ######################################

def parse_function(example_proto):
    feature_description = {
        'V0': tf.io.FixedLenFeature([], tf.string),
        'Vt': tf.io.FixedLenFeature([], tf.string),
        'M0': tf.io.FixedLenFeature([], tf.string),
        'Mt': tf.io.FixedLenFeature([], tf.string),
        'resx': tf.io.FixedLenFeature([], tf.string),
        'resy': tf.io.FixedLenFeature([], tf.string),
        'resz': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    V0 = tf.io.decode_raw(parsed_features['V0'], tf.float32)
    Vt = tf.io.decode_raw(parsed_features['Vt'], tf.float32)
    M0 = tf.io.decode_raw(parsed_features['M0'], tf.float32)
    Mt = tf.io.decode_raw(parsed_features['Mt'], tf.float32)
    resx = tf.io.decode_raw(parsed_features['resx'], tf.float32)
    resy = tf.io.decode_raw(parsed_features['resy'], tf.float32)
    resz = tf.io.decode_raw(parsed_features['resz'], tf.float32)
    V0 = tf.reshape(V0, [128, 128, 16, 1])
    Vt = tf.reshape(Vt, [128, 128, 16, 1])
    M0 = tf.reshape(M0, [128, 128, 16, 1])
    Mt = tf.reshape(Mt, [128, 128, 16, 1])
    resx = tf.reshape(resx, [128, 128, 16, 1])
    resy = tf.reshape(resy, [128, 128, 16, 1])
    resz = tf.reshape(resz, [128, 128, 16, 1])

    x = {'input_1': V0, 'input_2': Vt}
    y = tf.stack([V0, Vt, M0, Mt, resx, resy, resz], axis=0)

    return x,y

def load_tfrecord(filename):
    raw_dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = raw_dataset.map(parse_function)
    return parsed_dataset

##########################      Data Aug     ######################################

class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_batch = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", seed=None),
      tf.keras.layers.experimental.preprocessing.RandomRotation(1, seed=None),
      tf.keras.layers.experimental.preprocessing.RandomContrast(0.5, seed=None)
    ])

  def call(self, inputs, labels):
    ## unstack 2 batched inputs shape (batch, 128, 128, 16, 1) each in axis -2, and stack them back after augmentation
    ## unstack batched labels shape (batch, 5, 128, 128, 16, 1) each in axis -2 and -5, and stack them back after augmentation
    ## concat after unstacking to generate one big batch
    nz = labels.shape[-2]
      
    augmented = self.augment_batch(tf.concat(
      [tf.concat(tf.unstack(inputs['input_1'], axis=-2), axis=-1),
      tf.concat(tf.unstack(inputs['input_2'], axis=-2), axis=-1),
      tf.concat(tf.unstack(labels[:,0], axis=-2), axis=-1),
      tf.concat(tf.unstack(labels[:,1], axis=-2), axis=-1),
      tf.concat(tf.unstack(labels[:,2], axis=-2), axis=-1),
      tf.concat(tf.unstack(labels[:,3], axis=-2), axis=-1),
      tf.concat(tf.unstack(labels[:,4], axis=-2), axis=-1),
      tf.concat(tf.unstack(labels[:,5], axis=-2), axis=-1),
      tf.concat(tf.unstack(labels[:,6], axis=-2), axis=-1)], 
      axis=-1
    ))

    inputs['input_1'] = tf.expand_dims(tf.stack(tf.unstack(augmented[...,:nz], axis=-1), axis=-1), axis=-1)
    inputs['input_2'] = tf.expand_dims(tf.stack(tf.unstack(augmented[...,nz:2*nz], axis=-1), axis=-1), axis=-1)

    labels = tf.stack([
      tf.expand_dims(tf.stack(tf.unstack(augmented[...,2*nz:3*nz], axis=-1), axis=-1), axis=-1),
      tf.expand_dims(tf.stack(tf.unstack(augmented[...,3*nz:4*nz], axis=-1), axis=-1), axis=-1),
      tf.expand_dims(tf.stack(tf.unstack(augmented[...,4*nz:5*nz], axis=-1), axis=-1), axis=-1),
      tf.expand_dims(tf.stack(tf.unstack(augmented[...,5*nz:6*nz], axis=-1), axis=-1), axis=-1),
      tf.expand_dims(tf.stack(tf.unstack(augmented[...,6*nz:7*nz], axis=-1), axis=-1), axis=-1),
      tf.expand_dims(tf.stack(tf.unstack(augmented[...,7*nz:8*nz], axis=-1), axis=-1), axis=-1),
      tf.expand_dims(tf.stack(tf.unstack(augmented[...,8*nz:9*nz], axis=-1), axis=-1), axis=-1)
    ], axis=1)
                        
    return inputs, labels



dataset = load_tfrecord('data/training/trainingEDES_con_res.tfrecord')


# shuffle and batch
dataset = (
    dataset
    #.cache()
    #.shuffle(50)
    .batch(1)
    #.map(Augment())
    #.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

# filter if input_1 and input_2 are the same
dataset = dataset.filter(lambda x,y: tf.reduce_any(tf.not_equal(x['input_1'], x['input_2']))).take(1)

##########################      MODEL     ######################################

opt = CarMEN_options()

with strategy.scope():
    model = deep_strain_model.DeepStrain(Adam, opt=opt)
    netME = model.get_netME()

##########################      TRAINING     ######################################

start_time = time.time()

netME.fit(dataset, epochs=6000)

print("--- %s seconds ---" % (time.time() - start_time))

netME.save_weights("netME_weights_new_EDES_1_6000_reg_dice.h5")



 




