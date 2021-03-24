### To train CNN for Classification

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import optimizers, callbacks
import time, json, os
import matplotlib.pyplot as plt

AbsLoc = 'D:\dataset_root_directory'
train_data_dir = os.path.join(AbsLoc, 'data/train')
validation_data_dir = os.path.join(AbsLoc, 'data/validation')

# dimensions of our images.
img_width, img_height = 320, 320

nb_train_samples = 2340
nb_validation_samples = 260
epochs = 750
batch_size = 26

# initial
best_val_acc = 0

class checkpoint(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global best_val_acc
        best_model_name = 'ModelWithBestVal_acc.h5'
        now_val_acc = logs.get('val_acc')

        if now_val_acc > best_val_acc:
            best_val_acc = now_val_acc
            self.model.save(os.path.join(AbsLoc, 'data/' + best_model_name))  # save the model with best val_acc

            with open(os.path.join(AbsLoc, 'data/ModelWithBestVal_acc.txt'), 'w') as fw:
                json.dump('best_val_acc = ' + str(best_val_acc), fw)

        print('\n', 'Now best val_acc is: %f' % best_val_acc)


time_in = time.time()  # record using time start

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Nadam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
'''
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
'''
# No any augmentation
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

check = checkpoint()

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[check])

f1 = open(os.path.join(AbsLoc, 'data/train_history.txt'), 'w')
json.dump(hist.history, f1)
f1.close()

model.save(os.path.join(AbsLoc, 'data/cnn_model.h5'))

time_out = time.time()    # record using time end
print('\n', 'Time cost:', '\n', time_out-time_in)
