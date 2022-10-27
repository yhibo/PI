# Manuel A. Morales (moralesq@mit.edu)
# Harvard-MIT Department of Health Sciences & Technology  
# Athinoula A. Martinos Center for Biomedical Imaging

import typing as tp
import xdrlib

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                     Conv2D, Conv3D, PReLU, UpSampling2D,
                                     UpSampling3D)

from .dense_image_warp import dense_image_warp3d as warp


##############################################
#################### LOSSES ##################
##############################################
class Dice:
    """
    N-D dice for segmentation
    """
    def loss(self, y_true, y_pred):
        ndims = len(y_pred.get_shape().as_list()) - 2
        vol_axes = list(range(1, ndims+1))
        
        top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
        bottom = tf.reduce_sum(y_true + y_pred, vol_axes)

        div_no_nan = tf.math.divide_no_nan if hasattr(tf.math, 'divide_no_nan') else tf.div_no_nan
        dice = tf.reduce_mean(div_no_nan(top, bottom))
        return -dice
    
class Grad:
    """
    N-D gradient loss.
    loss_mult can be used to scale the loss value - this is recommended if
    the gradient is computed on a downsampled vector field (where loss_mult
    is equal to the downsample factor).
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            r = [d, *range(1, d), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss(self, _, y_pred):

        if self.penalty == 'l1':
            dif = [tf.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
        grad = tf.add_n(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad
    
##############################################
#################### LAYERS ##################
##############################################

def conv_bn(tensor,
            nf,
            ks,
            conv_dim=2,
            strides=1,
            padding='same',
            activation='leaky_relu',
            dw_separable=False,
            use_bn=True):
    """Implements the structure for making a 2D Conv followed with a
    Batchnormalization when use_bn is set to True, and then applying an
    activation function.

    Parameters
    ----------
    tensor : tensorflow or keras tensor
        input tensor
    nf : int
        number of feature to be used in the 3D Convolutional layer
    ks : int or tuple of ints
        kernel size used in the convolution
    strides : int or tuple of ints, optional
        The strides to be used in the 3D conv layer.
    padding : str, optional
        the padding to be used in the 3D conv layer.
    activation : str, optional
        the activation to be applied after the Batchnormalization layer if it
        is used or right after the 3D Conv layer otherwise.
    use_bn : bool, optional
        set if a Batchnormalization layer is used after the 3D conv layer.

    Returns
    -------
    A keras tensor which represents  a 3DConv (+BN) + Activation

    """
    conv_name = 'Conv%iD' % conv_dim
    use_bias = not use_bn
    if dw_separable:
        sep_conv = 'Separable' + conv_name
        ti = getattr(layers, sep_conv)(nf,
                                       kernel_size=ks,
                                       strides=strides,
                                       use_bias=use_bias,
                                       padding=padding)(tensor)
    else:
        ti = getattr(layers, conv_name)(nf,
                                        kernel_size=ks,
                                        strides=strides,
                                        use_bias=use_bias,
                                        padding=padding)(tensor)
    if use_bn:
        ti = layers.BatchNormalization()(ti)
    if activation.lower() == 'leaky_relu':
        ti = layers.LeakyReLU()(ti)
    else:
        ti = layers.Activation(activation.lower())(ti)
    return ti


def ChannelAttention(f, rate, axis, mlp_activation):
    #  max_pool = layers.GlobalMaxPool2D()(f)
    #  avg_pool = layers.GlobalAvgPool2D()(f)
    x1 = tf.reduce_max(f, axis=axis)
    x2 = tf.reduce_mean(f, axis=axis)
    c = f.shape[-1]  # Channel axis is -1
    if rate is None:
        rate = 2
    r = int(c // rate)
    if r == 0:
        r = 1
    # Create MLP
    mlp = models.Sequential()
    mlp.add(layers.Input(shape=(c, )))
    mlp.add(layers.Dense(r, use_bias=False))
    if mlp_activation.lower() == 'leaky_relu':
        mlp.add(layers.LeakyReLU())
    else:
        mlp.add(layers.Activation(mlp_activation.lower()))

    mlp.add(layers.Dense(c, use_bias=False))

    ti = mlp(x1) + mlp(x2)
    Mc = layers.Activation('sigmoid')(ti)
    # Expand dims
    if isinstance(axis, tuple) and len(axis) == 3:
        Mc = Mc[:, tf.newaxis, tf.newaxis, tf.newaxis]
    elif isinstance(axis, tuple) and len(axis) == 2:
        Mc = Mc[:, tf.newaxis, tf.newaxis]
    else:
        Mc = Mc[:, tf.newaxis]

    return Mc


def SpatialAttention(f1, conv_dim=2):
    x1 = tf.reduce_max(f1, axis=-1)  # Channel axis is -1
    x2 = tf.reduce_mean(f1, axis=-1)
    # Expand dims
    x1 = x1[..., tf.newaxis]  # x1 = tf.expand_dims(x1, axis=-1)
    x2 = x2[..., tf.newaxis]  # x2 = tf.expand_dims(x2, axis=-1)

    x = tf.concat([x1, x2], axis=-1)
    # Avoiding the use of Separable Convolution here
    conv_name = 'Conv%iD' % conv_dim
    Ms = getattr(layers, conv_name)(1, 3, padding='same',
                                    activation='sigmoid')(x)
    return Ms

def conv3D(tensor,
           nf,
           ks,
           strides=1,
           padding='same',
           activation='leaky_relu',
           dw_separable=False,
           use_bn=True):
    """Implements the structure for making a 3D Conv followed with a
    Batchnormalization when use_bn is set to True, and then applying an
    activation function.

    Parameters
    ----------
    tensor : tensorflow or keras tensor
        input tensor
    nf : int
        number of feature to be used in the 3D Convolutional layer
    ks : int or tuple of ints
        kernel size used in the convolution
    strides : int or tuple of ints, optional
        The strides to be used in the 3D conv layer.
    padding : str, optional
        the padding to be used in the 3D conv layer.
    activation : str, optional
        the activation to be applied after the Batchnormalization layer if it
        is used or right after the 3D Conv layer otherwise.
    use_bn : bool, optional
        set if a Batchnormalization layer is used after the 3D conv layer.

    Returns
    -------
    A keras tensor which represents  a 3DConv (+BN) + Activation

    """
    # NOTE: separable convolution is not implemented
    ti = conv_bn(tensor,
                 nf,
                 ks,
                 conv_dim=3,
                 strides=strides,
                 padding=padding,
                 activation=activation,
                 use_bn=use_bn,
                 dw_separable=False)
    return ti

def CBAM3D(tensor,
           nf,
           ks,
           rate=None,
           strides=1,
           padding='same',
           activation='leaky_relu',
           dw_separable=False,
           use_bn=True):
    """ Convolutional Block Attention Module Woo et al. 2018.

    Parameters
    ----------
    tensor : tensorflow or keras tensor
        input tensor
    nf : stridesint
        number of feature to be used in the 3D Convolutional layer
    ks : int or tuple of ints
        kernel size used in the convolution
    strides : int or tuple of ints, optional
        The strides to be used in the 3D conv layer.
    padding : str, optional
        the padding to be used in the 3D conv layer.
    activation : str, optional
        the activation to be applied after the Batchnormalization layer if it
        is used or right after the 3D Conv layer otherwise.
    use_bn : bool, optional
        set if a Batchnormalization layer is used after the 3D conv layer.

    Returns
    -------
    A keras tensor which represents  a 3DConv (+BN) + Activation

    """

    ti = conv3D(tensor,
                nf,
                ks,
                activation=activation,
                strides=strides,
                padding=padding,
                use_bn=use_bn,
                dw_separable=dw_separable)

    axis = (1, 2, 3)
    Mc = ChannelAttention(ti, rate, axis, 'relu')
    ti = ti * Mc  # Tf makes the broadcasting for us

    Ms = SpatialAttention(ti, conv_dim=3)
    ti = ti * Ms  # TF makes the broadcast for us

    # Residual Block (if spatial input is reduced it is needed a projection)
    nf_tensor = tensor.shape[-1]
    if strides > 1 or padding.lower() != 'same' or nf_tensor != nf:
        # Projection Conv with k=1 and no activation
        # Avoiding the use of Separable Convolution here
        rb = conv3D(tensor,
                    nf,
                    1,
                    activation='linear',
                    strides=strides,
                    padding=padding,
                    use_bn=use_bn)
    else:
        rb = tensor

    ti = rb + ti

    return ti

@tf.keras.utils.register_keras_serializable()
class ShareConv3D(tf.keras.layers.Layer):
    def __init__(self,
                 weights,
                 padding,
                 strides,
                 dilations,
                 use_bias=False,
                 **kwargs):
        super(ShareConv3D, self).__init__(**kwargs)
        #  self.weights = weights
        self.use_bias = use_bias
        self.w = weights[0]
        if self.use_bias:
            self.b = weights[1]
        self.padding = padding.upper()
        self.strides = strides
        self.dilations = dilations

    def call(self, inputs):
        ti = tf.nn.convolution(inputs,
                          self.w,
                          padding=self.padding,
                          strides=self.strides,
                          dilations=self.dilations)
        if self.use_bias:
            ti = ti + self.b
        return ti

    def get_config(self):
        config = super(ShareConv3D, self).get_config()
        if self.use_bias:
            config.update({'weights': [self.w.numpy(), self.b.numpy()]})
        else:
            config.update({'weights': [self.w.numpy()]})
        config.update({
            'padding': self.padding,
            'strides': self.strides,
            'dilations': self.dilations,
            'use_bias': self.use_bias,
        })
        return config

def KSAC3D(ti, nf, rates=[4, 8], activation='swish', dw_separable=False, id=[0]):
    """Original Kernel-Sharing Atrous Convolution block Huan et al.
    """

    id[0] += 1

    # 1x1 Conv.
    if dw_separable:
        # TODO: Not implemented throw and error
        t1 = layers.SeparableConv3D(nf,
                                    1,
                                    strides=1,
                                    padding='same',
                                    use_bias=False)(ti)
    else:
        t1 = layers.Conv3D(nf, 1, strides=1, padding='same',
                           use_bias=False)(ti)
    # Global Average Pool + Conv + BN + Activation + Upsampling
    # Expand dimensions (None,1,1,1,#)
    t2 = tf.reduce_mean(ti, axis=(1, 2, 3))
    t2 = t2[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
    if dw_separable:
        # TODO: Not implemented throw and error
        t2 = layers.SeparableConv3D(nf,
                                    1,
                                    strides=1,
                                    padding='same',
                                    use_bias=False)(t2)
    else:
        t2 = layers.Conv3D(nf, 1, strides=1, padding='same',
                           use_bias=False)(t2)

    t2 = layers.BatchNormalization()(t2)
    if activation.lower() == 'leaky_relu':
        t2 = layers.LeakyReLU()(t2)
    else:
        t2 = layers.Activation(activation.lower())(t2)
    t2 = layers.UpSampling3D(K.int_shape(ti)[1:4])(t2)
    # Sharing Conv. Layer
    cl = layers.Conv3D(nf,
                       3,
                       strides=1,
                       padding='same',
                       use_bias=False,
                       name='share%i_r1' % id[0])
    # Share Dilate convolution
    t = cl(ti)  # Dilation 1 + BN + Act
    t = layers.BatchNormalization()(t)
    if activation.lower() == 'leaky_relu':
        t = layers.LeakyReLU()(t)
    else:
        t = layers.Activation(activation.lower())(t)
    kst = [t]
    for r in rates:
        # 3D -> H,W,D
        strides = (1, ) + cl.strides + (1, )
        # factor to dilate for [N,H,W,D,C]
        #  dilations = (1, ) + (2**1, ) * 3 + (1, )
        dilations = (1, ) + (r, ) * 3 + (1, )
        t = ShareConv3D(cl.weights,
                        padding=cl.padding,
                        strides=strides,
                        dilations=dilations,
                        name='share%i_r%i' % (id[0], r))(ti)
        t = layers.BatchNormalization()(t)
        if activation.lower() == 'leaky_relu':
            t = layers.LeakyReLU()(t)
        else:
            t = layers.Activation(activation.lower())(t)
        kst.append(t)

    # Concatenation
    t_output = layers.Concatenate()([t1, t2] + kst)
    # Combine to fix into nf
    t_output = conv3D(t_output,
                      nf,
                      1,
                      activation=activation,
                      dw_separable=dw_separable)
    return t_output

def conv(Conv, layer_input, filters, kernel_size=3, strides=1, residual=False):
    """Convolution layer: Ck=Convolution-BatchNorm-PReLU"""
    dr = Conv(filters, kernel_size=kernel_size, strides=strides, padding='same')(layer_input)
    d  = BatchNormalization(momentum=0.5)(dr)  
    d  = PReLU()(d)
    
    if residual:
        return dr, d
    else:
        return d

def deconv(Conv, UpSampling, layer_input, filters, kernel_size=3, strides=1):
    """Deconvolution layer: CDk=Upsampling-Convolution-BatchNorm-PReLU"""
    u = UpSampling(size=strides)(layer_input)
    u = conv(Conv, u, filters, kernel_size=kernel_size, strides=1)
    return u

def encoder(Conv, layer_input, filters, kernel_size=3, strides=2):
    """Layers for 2D/3D network used during downsampling: CD=Convolution-BatchNorm-LeakyReLU"""
    d = conv(Conv, layer_input, filters, kernel_size=kernel_size, strides=1)
    dr, d = conv(Conv, d, filters, kernel_size=kernel_size, strides=strides, residual=True)
    d  = Conv(filters, kernel_size=kernel_size, strides=1, padding='same')(d)
    d  = Add()([dr, d])
    return d

def attention_KSAC_encoder(Conv, layer_input, filters, kernel_size=3, strides=2):
    """Layers for 2D/3D network used during downsampling: CD=Convolution-BatchNorm-LeakyReLU"""
    d  = CBAM3D(layer_input, filters, kernel_size, strides=1, padding='same', activation='leaky_relu', use_bn=True)
    dr, d = conv(Conv, d, filters, kernel_size=kernel_size, strides=strides, residual=True)
    d = KSAC3D(d, filters, rates=(2, 3))
    d  = Add()([dr, d])
    return d

def decoder(Conv, UpSampling, layer_input, skip_input, filters, kernel_size=3, strides=2):
    """Layers for 2D/3D network used during upsampling"""
    u = conv(Conv, layer_input, filters, kernel_size=1, strides=1)
    u = deconv(Conv, UpSampling, u, filters, kernel_size=kernel_size, strides=strides)
    u = Concatenate()([u, skip_input])
    u = conv(Conv, u, filters, kernel_size=kernel_size, strides=1)
    return u

def attention_KSAC_decoder(Conv, UpSampling, layer_input, skip_input, filters, kernel_size=3, strides=2):
    """Layers for 2D/3D network used during downsampling: CD=Convolution-BatchNorm-LeakyReLU"""
    u = KSAC3D(layer_input, filters, rates=(2, 3))
    u = deconv(Conv, UpSampling, u, filters, kernel_size=kernel_size, strides=strides)
    u = Concatenate()([u, skip_input])
    u  = CBAM3D(u, filters, kernel_size, strides=1, padding='same', activation='leaky_relu', use_bn=True)
    return u

def encoder_decoder(x, gf=64, nchannels=3, map_activation=None):
    
    if len(x.shape) == 5:
        Conv        = Conv3D
        UpSampling  = UpSampling3D
        strides     = (2,2,1)
        kernel_size = (3,3,1)
    elif len(x.shape) == 4:
        Conv        = Conv2D
        UpSampling  = UpSampling2D
        strides     = (2,2)
        kernel_size = (3,3)
            
    d1 = attention_KSAC_encoder(Conv, x,  gf*1, strides=strides, kernel_size=kernel_size)
    d2 = encoder(Conv, d1, gf*2, strides=strides, kernel_size=kernel_size)
    d3 = encoder(Conv, d2, gf*4, strides=strides, kernel_size=kernel_size)
    d4 = encoder(Conv, d3, gf*8, strides=strides, kernel_size=kernel_size)
    d5 = encoder(Conv, d4, gf*8, strides=strides, kernel_size=kernel_size)
    d6 = encoder(Conv, d5, gf*8, strides=strides, kernel_size=kernel_size)
    d7 = encoder(Conv, d6, gf*8, strides=strides, kernel_size=kernel_size)
    
    u1 = decoder(Conv, UpSampling, d7, d6, gf*8, strides=strides, kernel_size=kernel_size)
    u2 = decoder(Conv, UpSampling, u1, d5, gf*8, strides=strides, kernel_size=kernel_size)
    u3 = decoder(Conv, UpSampling, u2, d4, gf*8, strides=strides, kernel_size=kernel_size)
    u4 = decoder(Conv, UpSampling, u3, d3, gf*4, strides=strides, kernel_size=kernel_size)
    u5 = decoder(Conv, UpSampling, u4, d2, gf*2, strides=strides, kernel_size=kernel_size)
    u6 = attention_KSAC_decoder(Conv, UpSampling, u5, d1, gf*1, strides=strides, kernel_size=kernel_size)

    u7 = UpSampling(size=strides)(u6)
    u7 = Conv(nchannels, kernel_size=kernel_size, strides=1, padding='same', activation=map_activation)(u7)    
    
    return u7

class CarSON():
    """Cardiac Segmentation Network."""
    
    def __init__(self, optimizer, opt):
        self.opt = opt
        self.optimizer = optimizer

    def compile_model(self, model):
        if not self.opt.isTrain:
            model.compile(loss=None, 
                          optimizer=self.optimizer(learning_rate=0))
        else:
            model.compile(loss=self.opt.criterion_netS, 
                          optimizer=self.optimizer(learning_rate=self.opt.netS_lr))

    def get_model(self):
        V = keras.Input(shape=self.opt.image_shape) 
        M = encoder_decoder(V, nchannels=self.opt.nlabels, map_activation='softmax')
        
        model = keras.Model(inputs=V, outputs=M)
        self.compile_model(model)
        
        return model
    
##############################################
################## NETWORKS ##################
############################################## 
class CarMEN():
    """Cardiac Motion Estimation Network."""
    
    def __init__(self, optimizer, opt):
        self.opt = opt
        self.optimizer = optimizer
        
    def compile_model(self, model, loss_w=None):
        if not self.opt.isTrain:
            model.compile(loss=None, 
                          optimizer=self.optimizer(learning_rate=0))
        else:
            model.compile(loss=self.opt.criterion_netME, 
                          optimizer=self.optimizer(learning_rate=self.opt.netME_lr))
    def get_model(self):    
        V_0 = keras.Input(shape=self.opt.volume_shape) 
        V_t = keras.Input(shape=self.opt.volume_shape)
        V   = keras.layers.Concatenate(axis=-1)([V_0, V_t])

        u = encoder_decoder(V, nchannels=3, map_activation=None)
                       
        if not self.opt.isTrain:
            model = keras.Model(inputs=[V_0, V_t], outputs=u)
            self.compile_model(model)
        else:
            model = keras.Model(inputs=[V_0, V_t], outputs=u)
            self.compile_model(model)
            
        return model