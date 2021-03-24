
from keras.preprocessing.image import ImageDataGenerator
import time, json, os


# dimensions of our images.
img_width, img_height = 320, 320

AbsLoc = 'D:\GAN\dcgan'
# top_model_weights_path = 'fc_model.h5'
data_dir = os.path.join(AbsLoc, 'data/train/norm')

batch_size = 26

datagen = ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             fill_mode='reflect',
                             horizontal_flip=True,
                             vertical_flip=True)

generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    save_to_dir=os.path.join(AbsLoc, 'data/preview'), save_prefix='more_norm', save_format='png')


i = 1
for batch in generator:
    i += 1
    if i > 45:
        break

