from __future__ import print_function
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Nadam
from PIL import Image
import glob, numpy
import matplotlib.pyplot as plt
import time
import sys
import numpy as np

save_model_path = "saved_model_folder/dcgan_generator.h5"
rnd_len = 100  # random vector length for generator

############################### Build DCGAN models
class DCGAN():
    def __init__(self):
        self.img_rows = 320
        self.img_cols = 320
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Nadam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(rnd_len,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (rnd_len,)

        model = Sequential()
        model.add(Dense(256 * 20 * 20, activation="relu", input_shape=noise_shape))
        model.add(Reshape((20, 20, 256)))
        model.add(BatchNormalization(momentum=0.5))
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.5))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.5))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.5))
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(3, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, input_shape=img_shape, padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())

        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=12, save_interval=50):

############################### Load the dataset
        image_list = []
        for filename in glob.glob('input_dataset_folder/*.png'):
            im = Image.open(filename)
            reim = im.resize((320, 320), Image.ANTIALIAS)
            image_list.append(reim)

        image_stack = numpy.asarray(image_list[0].convert('RGB'))  # first image in stack
        image_stack = image_stack.transpose(2, 1, 0)  # (x,y,3) to (3,x,y), if needed
        image_stack = np.expand_dims(image_stack, axis=0)  # (3,x,y) to (1,3,x,y)

        for IM in image_list[1:]:   # rest images in stack
            IM = numpy.asarray(IM.convert('RGB'))
            IM = IM.transpose(2, 1, 0)  # (x,y,3) to (3,x,y), if needed
            IM = np.expand_dims(IM, axis=0)  # (3,x,y) to (1,3,x,y)
            image_stack = numpy.concatenate((IM, image_stack), axis=0)  # (1,3,x,y) to (n,3,x,y)
            
            
############################### Train DCGAN
        X_train = image_stack
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # Rescale -1 to 1

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Sample noise and generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, rnd_len))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, rnd_len))

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):  # print samples from generator
        r, c = 3, 3
        noise = np.random.normal(0, 1, (r * c, rnd_len))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(12, 12))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("output_images_folder/samples_%d.png" % epoch, dpi=100)
        plt.close()
        self.generator.save(save_model_path)


if __name__ == '__main__':  # run
    dcgan = DCGAN()
    time_in = time.time()   # record using time start
    dcgan.train(epochs=100001, batch_size=30, save_interval=200)
    time_out = time.time()  # record using time end
    print('\n', 'Time cost:', '\n', time_out-time_in)
