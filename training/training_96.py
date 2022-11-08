import argparse
import time

import tensorflow as tf
from models.dense_image_warp import dense_image_warp3d as warp
from models.networks_96 import Dice, Grad
from models.networks_96 import CarMEN
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

#tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

##########################       Args Parser       #####################################

parser = argparse.ArgumentParser(description='Train netME')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--loss', type=str, default='dice', help='Loss function')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.01, 0.5, 0.1, 0.01, 10, 1], help='Loss weights')
parser.add_argument('--shuffle_buffer_size', type=int, default=100, help='Shuffle buffer size')

##########################      Model Options     ######################################

class JD3D:

  def get_jacobian_det(self, y):
      dx = y[:, 1:, 1:, 1:, :] - y[:, :-1, 1:, 1:, :]
      dy = y[:, 1:, 1:, 1:, :] - y[:, 1:, :-1, 1:, :]
      dz = y[:, 1:, 1:, 1:, :] - y[:, 1:, 1:, :-1, :]

      du = tf.stack([dx[...,0]+1, dy[...,0], dz[...,0]], axis=-1)
      dv = tf.stack([dx[...,1], dy[...,1]+1, dz[...,1]], axis=-1)
      dw = tf.stack([dx[...,2], dy[...,2], dz[...,2]+1], axis=-1)

      return tf.linalg.det(tf.stack([du, dv, dw], axis=-1))
  
  def loss(self, _, y_pred):
      jd = self.get_jacobian_det(y_pred)
      neg_jac = 0.5 * (tf.abs(jd) - jd)
      return tf.reduce_mean(neg_jac)

class Losses:
  def __init__(self, lambda_i, lambda_a, lambda_s, lambda_c, lambda_j, lambda_z, loss='dice'):
    self.lambda_i = lambda_i
    self.lambda_a = lambda_a
    self.lambda_s = lambda_s
    self.lambda_c = lambda_c
    self.lambda_j = lambda_j
    self.lambda_z = lambda_z
    self.loss_anatomical = loss
        
  ##### LOSS 1 #####
  @tf.function
  def loss(self, y_true, y_pred):

    u = y_pred
    V_0 = tf.expand_dims(y_true[:,0], axis=-1)
    V_t = tf.expand_dims(y_true[:,1], axis=-1)
    M_0 = tf.expand_dims(y_true[:,2], axis=-1)
    M_t = tf.expand_dims(y_true[:,3], axis=-1)
    resx = tf.expand_dims(y_true[:,4], axis=-1)
    resy = tf.expand_dims(y_true[:,5], axis=-1)
    resz = tf.expand_dims(y_true[:,6], axis=-1)


    V_0_pred = warp(V_t, u)
    M_0_pred = warp(M_t, u)


    # Intensity loss
    L_i = K.mean((V_0_pred - V_0)**2, axis=(1,2,3,4))

    # Anatomical loss
    if self.loss_anatomical == 'dice':
      dice = Dice()
      L_a = dice.loss(M_0, M_0_pred) + 1
    else:
      L_a = K.mean((M_0 - M_0_pred)**2, axis=(1,2,3,4))
      L_a = K.mean(L_a, axis=-1)

    # Smoothness loss
    res = tf.concat([resx, resy, resz], axis=-1)
    # z has different spacing
    diff_mult = tf.stack([
      tf.ones_like(resx),
      tf.math.divide_no_nan(resx, resy),
      tf.math.divide_no_nan(resx, resz) ** 2
    ], axis=0)
    grad = Grad(penalty='l2', diff_mult=diff_mult)
    L_s = grad.loss([],u*res)

    # Consistency loss
    L_c = K.mean(K.abs(K.mean(M_0, axis=(1,2,4)) - K.mean(M_0_pred, axis=(1,2,4))), axis=-1)

    # Penalize negative Jacobian determinants
    jd3d = JD3D()
    L_j = jd3d.loss([], u)

    # Penalize positive z gradient along z axis
    L_z = K.permute_dimensions(u[...,2], (3, 0, 1, 2))
    L_z = L_z[1:] - L_z[:-1]
    L_z = L_z + K.abs(L_z)
    L_z = K.mean(L_z, axis=(0,2,3))


    #L_c = K.mean((M_0-M_0_pred*M_0)**2, axis=(1,2,3,4))

    return (  self.lambda_i * L_i + self.lambda_a * L_a +
              self.lambda_s * L_s + self.lambda_c * L_c +
              self.lambda_j * L_j + self.lambda_z * L_z   )


class CarMEN_options:
    def __init__(self, **kwargs):
        self.isTrain = kwargs.get('isTrain', True)
        self.volume_shape = kwargs.get('volume_shape', (96, 96, 16, 1))
        self.criterion_netME = kwargs.get('loss', None)
        self.netME_lr = kwargs.get('lr', 1e-4)

############################################ DATA AUGMENTATION ############################################

class Center_of_Scalar:
    def __init__(self, shape):
      X, Y, Z = tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), tf.range(shape[2]), indexing='ij')
      self.mesh = tf.cast(tf.stack([X, Y, Z], axis=-1), tf.float32)

    def __call__(self, img):
      centers = tf.reduce_sum(self.mesh * img, axis=(1,2,3), keepdims=True) / tf.reduce_sum(img, axis=(1,2,3), keepdims=True)
      return centers

    def get_mesh(self):
      return self.mesh


class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=None, volume_shape=(96, 96, 16), **kwargs):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_batch = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", seed=seed),
      tf.keras.layers.experimental.preprocessing.RandomRotation(1, seed=seed, fill_mode='constant'),
      tf.keras.layers.experimental.preprocessing.RandomTranslation(0.05, 0.05, seed=seed, fill_mode='constant'),
    ])
    self.CoS = Center_of_Scalar(volume_shape)
    self.volume_shape = volume_shape

  def expand_z_image_flow(self, img, mask, flow):
    nx, ny, _ = self.volume_shape
    mean = tf.math.reduce_mean(img, axis=[1,2,3], keepdims=True)
    mean = tf.repeat(mean, nx, axis=1)
    mean = tf.repeat(mean, ny, axis=2)
    zeros = tf.zeros_like(flow)
    zeros = tf.reduce_sum(zeros, axis=-2, keepdims=True)
    img = tf.concat([mean, img, mean], axis=3)
    flow  = tf.concat([zeros, flow, zeros], axis=3)
    zeros = tf.reduce_sum(zeros, axis=-1, keepdims=True)
    mask = tf.concat([zeros, mask, zeros], axis=3)
    return img, mask, flow

  def random_warp_z(self, img, mask, flowmult):

    centersz = self.CoS(img)[...,2]
    mesh = self.CoS.get_mesh()

    flow = tf.stack([tf.zeros_like(img[...,0]),
                    tf.zeros_like(img[...,0]),
                    tf.ones_like(img[...,0])*mesh[...,2] - centersz], axis=-1)
    flow /= tf.reduce_max(tf.abs(flow), axis=(1,2,3,4), keepdims=True)
    flow = flow * tf.random.uniform([1,], minval=0, maxval=4, dtype=tf.float32)
    flow = tf.math.round(flow)
    flow = flow*flowmult

    img, mask, flow = self.expand_z_image_flow(img, mask, flow)

    return warp(img, flow)[..., 1:-1, :], warp(mask, flow)[..., 1:-1, :]

  def call(self, inputs, labels):
    ## unstack 2 batched inputs shape (batch, 96, 96, 16, 1) each in axis -2, and stack them back after augmentation
    ## unstack batched labels shape (batch, 5, 96, 96, 16, 1) each in axis -2 and -5, and stack them back after augmentation
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

    trgt_sz = tf.random.uniform(shape=[1,], minval=0.9, maxval=1.4) * 96
    trgt_sz = tf.cast(trgt_sz, tf.int32)[0]
    augmented = tf.image.resize_with_crop_or_pad(augmented, trgt_sz, trgt_sz)
    augmented = tf.image.resize(augmented, [96, 96])

    input1 = augmented[...,:nz]
    input2 = augmented[...,nz:2*nz]
    label4 = augmented[...,5*nz:6*nz]

    gamma = tf.random.uniform(shape=[1,], minval=0.7, maxval=1.3)[0]
    input1 = tf.image.adjust_gamma(input1, gamma=gamma)
    input2 = tf.image.adjust_gamma(input2, gamma=gamma)

    contrast = tf.random.uniform(shape=[1,], minval=0.7, maxval=1.3)[0]
    input1 = tf.image.adjust_contrast(input1, contrast_factor=contrast)
    input2 = tf.image.adjust_contrast(input2, contrast_factor=contrast)

    input1 = tf.expand_dims(tf.stack(tf.unstack(input1, axis=-1), axis=-1), axis=-1)
    input2 = tf.expand_dims(tf.stack(tf.unstack(input2, axis=-1), axis=-1), axis=-1)
    label4 = tf.expand_dims(tf.stack(tf.unstack(label4, axis=-1), axis=-1), axis=-1)

    preserveED = tf.reduce_sum(input1 - input2, axis=(1,2,3,4), keepdims=True)
    preserveED = tf.cast(tf.greater(tf.abs(preserveED), 1e-8), tf.float32)
    input2, label4 = self.random_warp_z(input2, label4, preserveED)

    axes = [1, 2, 3]
    mean = tf.reduce_mean(input1, axis=axes, keepdims=True)
    std = tf.math.reduce_std(input1, axis=axes, keepdims=True) + 1e-8
    input1 = (input1 - mean) / std
    mean = tf.reduce_mean(input2, axis=axes, keepdims=True)
    std = tf.math.reduce_std(input2, axis=axes, keepdims=True) + 1e-8
    input2 = (input2 - mean) / std

    label3 =  tf.expand_dims(tf.stack(tf.unstack(augmented[...,4*nz:5*nz], axis=-1), axis=-1), axis=-1)
    label3 = tf.cast(tf.greater(label3, 0.5), tf.float32)
    label4 = tf.cast(tf.greater(label4, 0.5), tf.float32)

    inputs['input_1'] = input1
    inputs['input_2'] = input2

    labels = tf.stack([
      input1,
      input2,
      label3,
      label4,
      tf.expand_dims(tf.stack(tf.unstack(augmented[...,6*nz:7*nz], axis=-1), axis=-1), axis=-1),
      tf.expand_dims(tf.stack(tf.unstack(augmented[...,7*nz:8*nz], axis=-1), axis=-1), axis=-1),
      tf.expand_dims(tf.stack(tf.unstack(augmented[...,8*nz:9*nz], axis=-1), axis=-1), axis=-1)
    ], axis=1)
                        
    return inputs, labels


############################################ DATA LOADING ############################################

def parse_function(example_proto):
    feature_description = {
        'V0': tf.io.FixedLenFeature([], tf.string),
        'Vt': tf.io.FixedLenFeature([], tf.string),
        'M0': tf.io.FixedLenFeature([], tf.string),
        'Mt': tf.io.FixedLenFeature([], tf.string),
        'M0_dilated': tf.io.FixedLenFeature([], tf.string),
        'resxy': tf.io.FixedLenFeature([], tf.string),
        'resz': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    V0 = tf.io.decode_raw(parsed_features['V0'], tf.float32)
    Vt = tf.io.decode_raw(parsed_features['Vt'], tf.float32)
    M0 = tf.io.decode_raw(parsed_features['M0'], tf.float32)
    Mt = tf.io.decode_raw(parsed_features['Mt'], tf.float32)
    M0_dilated = tf.io.decode_raw(parsed_features['M0_dilated'], tf.float32)
    resxy = tf.io.decode_raw(parsed_features['resxy'], tf.float32)
    resz = tf.io.decode_raw(parsed_features['resz'], tf.float32)
    V0 = tf.reshape(V0, [96, 96, 16, 1])
    Vt = tf.reshape(Vt, [96, 96, 16, 1])
    M0 = tf.reshape(M0, [96, 96, 16, 1])
    Mt = tf.reshape(Mt, [96, 96, 16, 1])
    M0_dilated = tf.reshape(M0_dilated, [96, 96, 16, 1])
    resxy = tf.reshape(resxy, [96, 96, 16, 1])
    resz = tf.reshape(resz, [96, 96, 16, 1])

    x = {'input_1': V0, 'input_2': Vt}
    y = tf.stack([V0, Vt, M0, Mt, M0_dilated, resxy, resz], axis=0)

    return x,y

def load_tfrecord(filename):
    raw_dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = raw_dataset.map(parse_function)
    return parsed_dataset


def main(args=None):

  args = parser.parse_args(args=args)

  BATCHSIZE = args.batch_size
  EPOCHS = args.epochs
  LR = args.lr
  LOSS = args.loss
  lambda_i = args.loss_weights[0]
  lambda_a = args.loss_weights[1]
  lambda_s = args.loss_weights[2]
  lambda_c = args.loss_weights[3] if len(args.loss_weights) > 3 else 0
  lambda_j = args.loss_weights[4] if len(args.loss_weights) > 4 else 0
  lambda_z = args.loss_weights[5] if len(args.loss_weights) > 5 else 0

  SHUFFLE_BUFFER_SIZE = args.shuffle_buffer_size

  L = Losses(lambda_i, lambda_a, lambda_s, lambda_c, lambda_j, lambda_z, LOSS)

  loss = L.loss

  dataset = load_tfrecord('data/training/trainingEDES_con_res_96.tfrecord')

  dataset = (
      dataset
      .cache()
      .shuffle(SHUFFLE_BUFFER_SIZE)
      .batch(BATCHSIZE)
      .map(Augment(volume_shape=(96, 96, 16)))
      .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      )

  ############################################ MODEL ############################################

  opt = CarMEN_options(isTrain=True, loss=loss, lr=LR, volume_shape=(96, 96, 16, 1))

  with strategy.scope():
      model = CarMEN(Adam ,opt)
      netME = model.get_model()

  ############################################ TRAINING ############################################

  start_time = time.time()

  netME.fit(dataset, epochs=EPOCHS)

  print("--- %s seconds ---" % (time.time() - start_time))

  ############################################ SAVING ############################################

  model_name = 'netME_CBAM_96'
  model_name += '_epochs' + str(EPOCHS)
  model_name += '_batch' + str(BATCHSIZE)
  model_name += '_lr' + str(LR)
  model_name += '_loss_' + str(LOSS)
  model_name += ('_loss_weights_' + str(lambda_i) + '_' + str(lambda_a) +
                '_' + str(lambda_s) + '_' + str(lambda_c) +
                '_' + str(lambda_j) + '_' + str(lambda_z))
  model_name += '.h5'

  netME.save(model_name)

if __name__ == '__main__':
  main()

