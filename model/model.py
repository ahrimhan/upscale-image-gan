from keras.layers import Input, Activation, Add, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from .reflectionpadding2d import ReflectionPadding2D
from .resnet import res_block
from .losses import wasserstein_loss, perceptual_loss

from keras.optimizers import Adam
import keras.backend.tensorflow_backend as K
import tensorflow as tf

import os
import datetime
import numpy as np
import tqdm
from PIL import Image


ngf = 16
ndf = 16
n_blocks_gen = 7
critic_updates = 5

def floatize_image(i_img):
    f_img = np.array(i_img)
    f_img = (f_img - 127.5) / 127.5
    return f_img

def integerize_image(f_img):
    img = f_img * 127.5 + 127.5
    return img.astype('uint8')

def load_images(path_lists, image_width, image_height, training_ratio):
    high_images = []
    low_images = []
    for path in path_lists:
        high_img = Image.open(path).resize((image_width, image_height))
        high_images.append(floatize_image(high_img))
        low_images.append(floatize_image(high_img.resize(
                (int(image_width/training_ratio), int(image_height/training_ratio))
            ).resize(
                (image_width, image_height)
            )))
    return np.array(high_images), np.array(low_images)
        

class UpscaleGAN:
    def __init__(self, image_width=480, image_height=240):
        self.image_shape = (image_height, image_width, 3)
        self.image_width = image_width
        self.image_height = image_height
        self.isCompiled = False
        self.isBuild = False
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
        self.batch_size = 0

    def __compile(self):

        self.discriminator = self.__build_discriminator()
        self.generator = self.__build_generator()



        d_optimizer = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        g_optimizer = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        #self.discriminator.compile(
        #        loss='binary_crossentropy',
        #        optimizer=d_optimizer, 
        #        metrics=['accuracy'])
        self.discriminator.compile(optimizer=d_optimizer, loss=wasserstein_loss)

        self.generator.compile(
                loss='binary_crossentropy', 
                optimizer=g_optimizer)
        #self.generator.compile(
        #        loss=loss,
        #        optimizer=g_optimizer,
        #        loss_weights=loss_weights)
        loss = [perceptual_loss, wasserstein_loss]
        loss_weights = [100, 1]


        self.discriminator.trainable = False

        z = Input(shape=self.image_shape)
        img = self.generator(z)
        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)
        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(inputs=z, outputs=[img, valid])
        #self.combined.compile(
        #        loss='binary_crossentropy', 
        #        optimizer=g_optimizer)
        self.combined.compile(
                loss=loss,
                optimizer=g_optimizer,
                loss_weights=loss_weights)
        self.isCompiled = True

    def __train_init(self, image_filepaths, batch_size=8, training_ratio=2):
        self.batch_size = batch_size
        self.image_filepaths = image_filepaths
        self.training_ratio = training_ratio
        self.true_value = np.ones((batch_size, 1))
        self.false_value = -np.ones((batch_size, 1))

        if not self.isCompiled:
            self.__compile()
            self.isCompiled = True

    def __train_one_epoch(self):
        d_losses = []
        g_losses = []

        image_files_array = np.array(self.image_filepaths)

        permutated_indexes = np.random.permutation(image_files_array.shape[0])

        for i in range(int(512 / self.batch_size)):
            batch_indexes = permutated_indexes[i*self.batch_size:(i+1)*self.batch_size]
            paths = image_files_array[batch_indexes]
            high_image_batch, low_image_batch = load_images(paths, self.image_width, self.image_height, self.training_ratio)
    
            generated_images = self.generator.predict(x=low_image_batch, batch_size=self.batch_size)
            for _ in range(critic_updates):
                d_loss_real = self.discriminator.train_on_batch(high_image_batch, self.true_value)
                d_loss_fake = self.discriminator.train_on_batch(generated_images, self.false_value)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)
            g_loss = self.combined.train_on_batch(low_image_batch, [high_image_batch, self.true_value])
            g_losses.append(g_loss)
        d_loss_mean = np.mean(d_losses)
        g_loss_mean = np.mean(g_losses)

        return d_loss_mean, g_loss_mean

    def train(self, image_filepaths, batch_size=8, training_ratio=2, epochs=16, epoch_callback=None):
        self.__train_init(image_filepaths, batch_size, training_ratio)
        for i in tqdm.tqdm(range(epochs)):
            d_loss, g_loss = self.__train_one_epoch()
            if epoch_callback:
                epoch_callback(self, i, d_loss, g_loss)

    def test(self, image_filepath, result_filepath, ratio):
        high_image_batch, low_image_batch = load_images([image_filepath], self.image_width, self.image_height, ratio)
        generated_images = self.generator.predict(x=low_image_batch, batch_size=1)

        generated = np.array([integerize_image(img) for img in generated_images])
        high_image_batch = integerize_image(high_image_batch)
        low_image_batch = integerize_image(low_image_batch)

        y = high_image_batch[0, :, :, :]
        x = low_image_batch[0, :, :, :]
        img = generated[0, :, :, :]
        output = np.concatenate((y, x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save(result_filepath)

    def save(self, epoch, d_loss, g_loss, output_path):
        self.generator.save_weights(os.path.join(output_path, 'generator_{}_{}.h5'.format(epoch, g_loss)), True)
        self.discriminator.save_weights(os.path.join(output_path, 'discriminator_{}_{}.h5'.format(epoch, d_loss)), True)

    def __build_generator(self):
        """Build generator architecture."""
        # Current version : ResNet block
        inputs = Input(shape=self.image_shape)

        x = ReflectionPadding2D((3, 3))(inputs)
        x = Conv2D(filters=ngf, kernel_size=(7, 7), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            x = Conv2D(filters=ngf*mult*2, kernel_size=(3, 3), strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        mult = 2**n_downsampling
        for i in range(n_blocks_gen):
            x = res_block(x, ngf*mult, use_dropout=True)

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            # x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=(3, 3), strides=2, padding='same')(x)
            x = UpSampling2D()(x)
            x = Conv2D(filters=int(ngf * mult / 2), kernel_size=(3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(filters=3, kernel_size=(7, 7), padding='valid')(x)
        x = Activation('tanh')(x)

        outputs = Add()([x, inputs])
        # outputs = Lambda(lambda z: K.clip(z, -1, 1))(x)
        outputs = Lambda(lambda z: z/2)(outputs)

        model = Model(inputs=inputs, outputs=outputs, name='Generator')
        return model


    def __build_discriminator(self):
        """Build discriminator architecture."""
        n_layers, use_sigmoid = 3, False
        inputs = Input(shape=self.image_shape)

        x = Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding='same')(inputs)
        x = LeakyReLU(0.2)(x)

        nf_mult, nf_mult_prev = 1, 1
        for n in range(n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
            x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)

        nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
        x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
        if use_sigmoid:
            x = Activation('sigmoid')(x)

        x = Flatten()(x)
        x = Dense(1024, activation='tanh')(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=x, name='Discriminator')
        return model

