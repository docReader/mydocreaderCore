from __future__ import print_function, division
from pickle import NONE
import scipy
import scipy.misc

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import load_model
from tqdm import tqdm
from glob import glob

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from datetime import datetime
import sys
import numpy as np
import pandas as pd
import os
import cv2
from glob import glob
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K

import subprocess as sp
import tensorflow as tf

def get_available_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

gpus = tf.config.list_physical_devices('GPU')
print('List of available GPUs', gpus)
limit_size = 1000

if gpus:
    available_gpu_mem = get_available_gpu_memory()
    print(available_gpu_mem)
    # limit_size = available_gpu_mem[0]/6
    
# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=limit_size)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    print('*** GPU limited for Tensorflow at {} MiB. ***'.format(limit_size))



class InstanceNormalization(Layer):
    
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DataLoader():
    def __init__(self, img_res=(256, 512)):
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False, is_random = True):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('./gan_data/%s/*' % data_type)

        if is_random: batch_images = np.random.choice(path, size=batch_size)
        else: batch_images = path
        
        imgs = []
        for img_path in batch_images:
            img = image.load_img(img_path, color_mode='grayscale', target_size=(256, 512))
            img = image.img_to_array(img).astype('float32')
            img = img / 255.0
            if not is_testing and is_random:

                if np.random.random() > 0.5:
                    img = np.fliplr(img)

              
            imgs.append(img)

        imgs = np.array(imgs)

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_typeA = "trainA" if not is_testing else "testA"
        data_typeB = "trainB" if not is_testing else "testB"
        path_A = glob('./gan_data/%s/*' % (data_typeA))
        path_B = glob('./gan_data/%s/*' % (data_typeB))
        
        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)
        
        

        for i in range(self.n_batches):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                
                img_A = image.load_img(img_A, color_mode='grayscale', target_size=(256, 512))
                img_A = image.img_to_array(img_A).astype('float32')
                img_A = img_A / 255.0
                
                img_B = image.load_img(img_B, color_mode='grayscale', target_size=(256, 512))
                img_B = image.img_to_array(img_B).astype('float32')
                img_B = img_B / 255.0

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img.resize(self.img_res)
        img = img/255.0
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return cv2.imread(path).astype(np.float)


class CycleGAN():
    # @profile
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 512
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch1 = int(self.img_rows / 2**4)
        patch2 = int(self.img_cols / 2**4)
        self.disc_patch = (patch1, patch2, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss (mainly to preserve color consistency)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        

        #img_A = img_A.reshape(1,256, 256,1)
        #img_B = img_B.reshape(1,256, 256,1)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
        
#         print(Model(d0,output_img).summary())

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)


def merge_imgs(imgs, res):

    target_height = res[0]
    target_width = res[1]

    result_img = []
    i = 0

    while len(result_img) == 0 or result_img.shape[0] < target_height:

        tmp_img = []

        while len(tmp_img) == 0 or tmp_img.shape[1] < target_width:

            if len(tmp_img) == 0:
                tmp_img = imgs[i]
            else:
                if tmp_img.shape[1] + imgs[i].shape[1] > target_width:
                    extra_width = tmp_img.shape[1] + imgs[i].shape[1] - target_width
                    cropped_img = imgs[i][:, extra_width:]
                    tmp_img = np.concatenate((tmp_img, cropped_img), axis=1)
                else:
                    tmp_img = np.concatenate((tmp_img, imgs[i]), axis=1)
            i += 1

        if len(result_img) == 0:
            result_img = tmp_img
        else:
            if result_img.shape[0] + tmp_img.shape[0] > target_height:
                extra_height = result_img.shape[0] + tmp_img.shape[0] - target_height
                cropped_img = tmp_img[extra_height:, :]
                result_img = np.concatenate((result_img, cropped_img), axis=0)
            else:
#                 local_time = time.time()
                result_img = np.concatenate((result_img, tmp_img), axis=0)
#                 print(time.time() - local_time)


    return result_img


def split_img(img, res):

    result_img = []
    split_height = res[0]
    split_width = res[1]

#     img = cv2.imread(img_loc, cv2.IMREAD_COLOR)

    source_height = img.shape[0]
    source_width = img.shape[1]

    if (source_height < split_height) or (source_width < split_width):
        print("Input image dimension is less than " + str(split_height) + " x " + str(split_width) )

    ht = 0
    row = 0

    while ht < source_height:
        wd = 0
        col = 0
        while wd < source_width:
            tmp_img = img

            if ht + split_height > source_height:
                diff = (ht + split_height) - source_height
                ht -= diff

            if wd + split_width > source_width:
                diff = (wd + split_width) - source_width
                wd -= diff

            img = img[ht:ht + split_height, wd:wd + split_width]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img=image.img_to_array(img).astype('float32')
            #img=img.reshape(256, 512,1)
            image_list = np.zeros((1, 256, 512, 1))
            image_list[0] = img
            
            img = gan.g_AB(image_list)*255.0
            
            if type(result_img) == list:
                result_img = img
            else:
                result_img = np.concatenate((result_img, img), axis = 0)

            img = tmp_img
            wd += split_width
            col += 1
        ht += split_height
        row += 1

    return result_img



gan = CycleGAN()

# @profile
def cycle_gan_memory():
    gan.g_AB = load_model(
        './src/auto_noise/cycleGanModels/epochs-30/g_AB.h5', 
        custom_objects={'InstanceNormalization':InstanceNormalization},
        compile=False,
    )

def correct_noise_cycleGan(img):
    cycle_gan_memory()
    split_images = split_img(img, [256, 512])

    recons_img = merge_imgs(split_images, img.shape)

    recons_img = cv2.cvtColor(recons_img,cv2.COLOR_GRAY2RGB)
    return recons_img

