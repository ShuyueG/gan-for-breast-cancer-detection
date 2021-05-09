import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications, optimizers, callbacks
import time, json, os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# dimensions of our images.
img_width, img_height = 256, 512

AbsLoc = 'D:\dataset_root_directory'
train_data_dir = os.path.join(AbsLoc, r'data/train')
validation_data_dir = os.path.join(AbsLoc, r'data/validation')
nb_train_samples = 360
nb_validation_samples = 90
epochs = 1500
batch_size = 15

# initial
best_val_acc = 0


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / (2**16-1))

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open(os.path.join(AbsLoc, 'data/bottleneck_features_train.npy'), 'wb'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open(os.path.join(AbsLoc, 'data/bottleneck_features_validation.npy'), 'wb'), bottleneck_features_validation)

def plot_roc(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.savefig(os.path.join(AbsLoc, 'data/ModelWithBestVal_acc_ROC.png'))  # save ROC fig
    

class checkpoint(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global best_val_acc
        best_model_name = 'ModelWithBestVal_acc.h5'
        now_val_acc = logs.get('val_acc')
        
        if now_val_acc > best_val_acc:
            best_val_acc = now_val_acc
            self.model.save(os.path.join(AbsLoc, 'data/' + best_model_name))  # save the model
            
            y_score = self.model.predict(test_data)
            fpr, tpr, _ = roc_curve(test_labels, y_score)
            roc_auc = auc(fpr, tpr)
            plot_roc(fpr, tpr, roc_auc) # save ROC fig
            with open(os.path.join(AbsLoc, 'data/ModelWithBestVal_acc.txt'), 'w') as fw:
                json.dump('best_val_acc = ' + str(best_val_acc), fw)
                fw.write('\n')
                json.dump('roc_auc = ' + str(roc_auc), fw)
                fw.write('\n')
                json.dump('fpr = ' + str(fpr), fw)
                fw.write('\n')
                json.dump('tpr = ' + str(tpr), fw)
                
        print('\n', 'Now best val_acc is: %f' % best_val_acc)
    
def train_top_model():
    train_data = np.load(open(os.path.join(AbsLoc, 'data/bottleneck_features_train.npy'), 'rb'))
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
	
    model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Nadam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
              metrics=['accuracy'])
    check = checkpoint()
    hist = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(test_data, test_labels),
              callbacks=[check])

    f1 = open(os.path.join(AbsLoc, 'data/bottleneck_fc_model.txt'), 'w')
    json.dump(hist.history, f1)
    f1.close()

#  get features from pre-trained VGG16 network
save_bottlebeck_features()
test_data = np.load(open(os.path.join(AbsLoc, 'data/bottleneck_features_validation.npy'), 'rb'))
test_labels = np.array(
    [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

#  train top model
train_top_model()




