import os
from re import M
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from models.dense_image_warp import dense_image_warp3d as warp
from models.networks import *
from models import deep_strain_model
from tensorflow.keras.optimizers import Adam


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

    print("y_true shape: ", y_true.shape)
    print("y_pred shape: ", y_pred.shape)

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


##########################      DATA     ######################################

DATA_FOLDER = 'dataset'

V0_files = [f for f in os.listdir(DATA_FOLDER) if f.startswith('Vo')]
Vt_files = [f for f in os.listdir(DATA_FOLDER) if f.startswith('Vt')]
M0_files = [f for f in os.listdir(DATA_FOLDER) if f.startswith('Mo')]
Mt_files = [f for f in os.listdir(DATA_FOLDER) if f.startswith('Mt')]
res_files = [f for f in os.listdir(DATA_FOLDER) if f.startswith('res')]


class DataGenerator(keras.utils.Sequence):

    def __init__(self, V0_files, Vt_files, M0_files, Mt_files, res_files):
        """Constructor can be expanded,
           with batch size, dimentation etc.
        """
        self.V0_file_list = V0_files
        self.Vt_file_list = Vt_files
        self.M0_file_list = M0_files
        self.Mt_file_list = Mt_files
        self.res_file_list = res_files
        self.on_epoch_end()

    def __len__(self):
      'Take all batches in each iteration'
      return int(len(self.V0_file_list))

    def __getitem__(self, index):
      'Get next batch'
      # Generate indexes of the batch
      indexes = self.indexes[index:(index+1)]

      # single file
      file_list_temp = [[self.V0_file_list[k],
                        self.Vt_file_list[k],
                        self.M0_file_list[k],
                        self.Mt_file_list[k],
                        self.res_file_list[k]] for k in indexes]

      # Set of X_train and y_train
      X, y = self.__data_generation(file_list_temp)

      return X, y

    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.V0_file_list))

    def __data_generation(self, file_list_temp):
      'Generates data containing batch_size samples'
      data_loc = DATA_FOLDER
      # Generate data
      for ID in file_list_temp:

          V0_file_path = os.path.join(data_loc, ID[0])
          Vt_file_path = os.path.join(data_loc, ID[1])
          M0_file_path = os.path.join(data_loc, ID[2])
          Mt_file_path = os.path.join(data_loc, ID[3])
          res_file_path = os.path.join(data_loc, ID[4])

          # Store sample
          X = [np.load(V0_file_path), np.load(Vt_file_path)]

          # Store class
          y = np.concatenate((np.array(X)[0],
                             np.array(X)[1],
                             np.load(M0_file_path),
                             np.load(Mt_file_path),
                             np.load(res_file_path)), axis=0)[..., np.newaxis]
          print("y shape: ", y.shape)
      return X, y


tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
opt = CarMEN_options()

with strategy.scope():
    model = deep_strain_model.DeepStrain(Adam, opt=opt)
    netME = model.get_netME()


# ====================
# train set
# ====================

training_generator = DataGenerator(V0_files, Vt_files, M0_files, Mt_files, res_files)

hst = netME.fit_generator(generator=training_generator, 
                           epochs=1, 
                           use_multiprocessing=True,
                           max_queue_size=32)