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


##########################      Model Options     ######################################

@tf.function
def criterion_netME(y_true, y_pred):

    u = y_pred
    V_0 = tf.expand_dims(y_true[:,0], axis=-1)
    V_t = tf.expand_dims(y_true[:,1], axis=-1)
    M_0 = tf.expand_dims(y_true[:,2], axis=-1)
    M_t = tf.expand_dims(y_true[:,3], axis=-1)
    res = tf.expand_dims(y_true[:,4], axis=-1)

    V_0_pred = warp(V_t, u)

    M_t_split = tf.split(M_t, M_t.shape[-1], -1)
    M_0_pred  = K.concatenate([warp(K.cast(mt, K.dtype(V_t)), u) for mt in M_t_split], -1)
    M_0_pred = tf.round(M_0_pred)
    # M_0_pred  = keras.activations.softmax(M_0_pred)

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


##########################      DATA     ######################################

def parse_function(example_proto):
    feature_description = {
        'V0': tf.io.FixedLenFeature([], tf.string),
        'Vt': tf.io.FixedLenFeature([], tf.string),
        'M0': tf.io.FixedLenFeature([], tf.string),
        'Mt': tf.io.FixedLenFeature([], tf.string),
        'res': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    V0 = tf.io.decode_raw(parsed_features['V0'], tf.float32)
    Vt = tf.io.decode_raw(parsed_features['Vt'], tf.float32)
    M0 = tf.io.decode_raw(parsed_features['M0'], tf.float32)
    Mt = tf.io.decode_raw(parsed_features['Mt'], tf.float32)
    res = tf.io.decode_raw(parsed_features['res'], tf.float32)
    V0 = tf.reshape(V0, [128, 128, 16, 1])
    Vt = tf.reshape(Vt, [128, 128, 16, 1])
    M0 = tf.reshape(M0, [128, 128, 16, 1])
    Mt = tf.reshape(Mt, [128, 128, 16, 1])
    res = tf.reshape(res, [128, 128, 16, 1])

    x = {'input_1': V0, 'input_2': Vt}
    y = tf.stack([V0, Vt, M0, Mt, res], axis=0)

    return x,y

def load_tfrecord(filename):
    raw_dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = raw_dataset.map(parse_function)
    return parsed_dataset

dataset = load_tfrecord('data/training/training.tfrecord')

# shuffle and batch
dataset = dataset.shuffle(50).batch(5)

##########################      MODEL     ######################################

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
opt = CarMEN_options()

with strategy.scope():
    model = deep_strain_model.DeepStrain(Adam, opt=opt)
    netME = model.get_netME()

##########################      TRAINING     ######################################

start_time = time.time()

netME.fit(dataset, epochs=100)

print("--- %s seconds ---" % (time.time() - start_time))

netME.save_weights("netME_weights_new.h5")









