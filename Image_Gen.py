### Image Augmentation by GAN

import keras
import os
import numpy as np
import matplotlib.pyplot as plt

AbsLoc = 'D:\dataset_root_directory'
model_location = os.path.join(AbsLoc, "generator_name.h5")  # GAN model

# Generate images
Gen_Num = 2340
rnd_len = 100
model = keras.models.load_model(model_location)
noise = np.random.normal(0, 1, (Gen_Num, rnd_len))
gen_imgs = model.predict(noise)

# Rescale images 0 - 1
gen_imgs = 0.5 * gen_imgs + 0.5

#  Save img files
for i in range(gen_imgs.shape[0]):
    img_save_loc = os.path.join(AbsLoc, "GAN_images/gen_%d.png" % (i + 1))
    # plt.imsave(img_save_loc, gen_imgs[i, :, :, :])  # for RGB imgs
    plt.imsave(img_save_loc, gen_imgs[i, :, :, 0], cmap ='gray')   # for gray imgs
