import tensorflow as tf
from tensorflow.keras import backend as K
from models.dense_image_warp import dense_image_warp3d as warp
from models.networks import *
from models import deep_strain_model
from tensorflow.keras.optimizers import Adam
import time
import argparse

#tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

##########################       Parse Args       ######################################

parser = argparse.ArgumentParser(description='Train netME')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--loss', type=str, default='MSE', help='Loss function')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.01, 0.5, 0.1, 0], help='Loss weights')
parser.add_argument('--shuffle_buffer_size', type=int, default=100, help='Shuffle buffer size')

args = parser.parse_args()

BATCHSIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
LOSS = args.loss
lambda_i = args.loss_weights[0]
lambda_a = args.loss_weights[1]
lambda_s = args.loss_weights[2]
if len(args.loss_weights) == 4:
    lambda_c = args.loss_weights[3]
SHUFFLE_BUFFER_SIZE = args.shuffle_buffer_size

##########################      Model Options     ######################################

##### LOSS 1 #####
@tf.function
def loss_dice(y_true, y_pred):

    u = y_pred
    V_0 = tf.expand_dims(y_true[:,0], axis=-1)
    V_t = tf.expand_dims(y_true[:,1], axis=-1)
    M_0 = tf.expand_dims(y_true[:,2], axis=-1)
    M_t = tf.expand_dims(y_true[:,3], axis=-1)
    resx = tf.expand_dims(y_true[:,4], axis=-1)
    resy = tf.expand_dims(y_true[:,5], axis=-1)
    resz = tf.expand_dims(y_true[:,6], axis=-1)

    V_0_pred = warp(V_t, u)

    M_0_pred0 = warp(K.cast(M_t==0, 'float32'), u)
    M_0_pred1 = warp(K.cast(M_t==1, 'float32'), u)
    M_0_pred2 = warp(K.cast(M_t==2, 'float32'), u)
    M_0_pred3 = warp(K.cast(M_t==3, 'float32'), u)

    dice = Dice()
    grad = Grad(penalty='l2')

    # Intensity loss
    L_i = K.mean(abs(V_0_pred - V_0), axis=(1,2,3,4))

    # Anatomical loss
    L_a = dice.loss(K.cast(M_0==0, 'float32'), M_0_pred0)
    L_a += dice.loss(K.cast(M_0==1, 'float32'), M_0_pred1)
    L_a += dice.loss(K.cast(M_0==2, 'float32'), M_0_pred2)
    L_a += dice.loss(K.cast(M_0==3, 'float32'), M_0_pred3)
    L_a /= 4

    # Smoothness loss
    res = tf.concat([resx, resy, resz], axis=-1)
    L_s = grad.loss([],u*res)

    return lambda_i * L_i + lambda_a * L_a + lambda_s * L_s + lambda_a

##### LOSS 2 #####
@tf.function
def loss_MSE(y_true, y_pred):

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

    grad = Grad(penalty='l2')

    # Intensity loss
    L_i = K.mean(abs(V_0_pred - V_0), axis=(1,2,3,4))

    # Anatomical loss
    L_a = K.mean((M_0_pred - M_0)**2, axis=(1,2,3,4))

    # Smoothness loss
    res = tf.concat([resx, resy, resz], axis=-1)
    L_s = grad.loss([],u*res)

    return lambda_i * L_i + lambda_a * L_a + lambda_s * L_s

##### LOSS 3 #####
@tf.function
def loss_new(y_true, y_pred):

    u = y_pred
    V_0 = tf.expand_dims(y_true[:,0], axis=-1)
    V_t = tf.expand_dims(y_true[:,1], axis=-1)
    M_0 = tf.expand_dims(y_true[:,2], axis=-1)
    M_t = tf.expand_dims(y_true[:,3], axis=-1)
    resx = tf.expand_dims(y_true[:,4], axis=-1)
    resy = tf.expand_dims(y_true[:,5], axis=-1)
    resz = tf.expand_dims(y_true[:,6], axis=-1)

    V_0_pred = warp(V_t, u)

    M_0_pred0 = warp(K.cast(M_t==0, 'float32'), u)
    M_0_pred1 = warp(K.cast(M_t==1, 'float32'), u)
    M_0_pred2 = warp(K.cast(M_t==2, 'float32'), u)
    M_0_pred3 = warp(K.cast(M_t==3, 'float32'), u)

    dice = Dice()
    grad = Grad(penalty='l2')

    # Intensity loss
    L_i = K.mean(abs(V_0_pred - V_0), axis=(1,2,3,4))

    # Anatomical loss
    L_a = dice.loss(K.cast(M_0==0, 'float32'), M_0_pred0)
    L_a += dice.loss(K.cast(M_0==1, 'float32'), M_0_pred1)
    L_a += dice.loss(K.cast(M_0==2, 'float32'), M_0_pred2)
    L_a += dice.loss(K.cast(M_0==3, 'float32'), M_0_pred3)
    L_a /= 4

    # Smoothness loss
    res = tf.concat([resx, resy, resz], axis=-1)
    L_s = grad.loss([],u*res)

    # Consistency loss
    L_c = K.mean(K.abs(K.mean(K.cast(M_0==0, 'float32'), axis=(0,1,2,4)) - K.mean(K.cast(M_0_pred0>0.5, 'float32'), axis=(0,1,2,4))))
    L_c += K.mean(K.abs(K.mean(K.cast(M_0==1, 'float32'), axis=(0,1,2,4)) - K.mean(K.cast(M_0_pred1>0.5, 'float32'), axis=(0,1,2,4))))
    L_c += K.mean(K.abs(K.mean(K.cast(M_0==2, 'float32'), axis=(0,1,2,4)) - K.mean(K.cast(M_0_pred2>0.5, 'float32'), axis=(0,1,2,4))))
    L_c += K.mean(K.abs(K.mean(K.cast(M_0==3, 'float32'), axis=(0,1,2,4)) - K.mean(K.cast(M_0_pred3>0.5, 'float32'), axis=(0,1,2,4))))
    L_c /= 4

    return lambda_i * L_i + lambda_a * L_a + lambda_s * L_s + lambda_a + lambda_c * L_c


# loss dictionary
loss_dict = {'dice': loss_dice, 'MSE': loss_MSE, 'new': loss_new}

class CarMEN_options:
    def __init__(self):
        self.isTrain = True
        self.volume_shape = (128, 128, 16, 1)
        self.criterion_netME = loss_dict[LOSS]
        self.netME_lr = LR

############################################ DATA AUGMENTATION ############################################

class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_batch = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", seed=None),
      tf.keras.layers.experimental.preprocessing.RandomRotation(1, seed=None),
      tf.keras.layers.experimental.preprocessing.RandomContrast(0.5, seed=None),
      tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2, seed=None)
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


############################################ DATA LOADING ############################################

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


dataset = load_tfrecord('data/training/trainingEDES_con_res.tfrecord')

dataset = (
    dataset
    .cache()
    .shuffle(SHUFFLE_BUFFER_SIZE)
    .batch(BATCHSIZE)
    .map(Augment())
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

############################################ MODEL ############################################

opt = CarMEN_options()

with strategy.scope():
    model = deep_strain_model.DeepStrain(Adam, opt=opt)
    netME = model.get_netME()

############################################ TRAINING ############################################

start_time = time.time()

netME.fit(dataset, epochs=EPOCHS)

print("--- %s seconds ---" % (time.time() - start_time))

############################################ SAVING ############################################

model_name = 'netME'
model_name += '_epochs' + str(EPOCHS)
model_name += '_batch' + str(BATCHSIZE)
model_name += '_lr' + str(LR)
model_name += '_loss_' + str(LOSS)
model_name += '_loss_weights_' + str(lambda_i) + '_' + str(lambda_a) + '_' + str(lambda_s)
if len(args.loss_weights) == 4:
    model_name += '_' + str(lambda_c)
model_name += '.h5'

netME.save_weights(model_name)




