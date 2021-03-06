# from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, MaxoutDense
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
import h5py
from weight_loading import weight_loading

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_batch = []
        self.val_losses_epoch = []
        self.acc = []
        def on_batch_end(self, batch, logs={}):
            self.losses_batch.append(logs.get('loss'))
        def on_epoch_end(self, batch, logs={}):
            self.val_losses_epoch.append(logs.get('val_loss'))
            self.acc.append(logs.get('acc'))


batch_size = 4096
nb_epoch = 100
data_augmentation = True
img_rows, img_cols = 24, 24
nb_classes = 2

with h5py.File('/data4/gopal.sharma/datasets/deep_text/text-non-text/train_text_non_text_pos.h5', 'r') as hf:
    print('List of arrays in this file: \n', hf.keys())
    data = hf.get('X')
    X_pos = np.array(data)
    data = hf.get('y')
    y_pos = np.array(data, dtype = np.int8)
X_pos = np.expand_dims(X_pos, axis = 1)
y_pos = np.expand_dims(y_pos, axis = 1)
print ((y_pos.shape, y_pos[0]))

with h5py.File('/data4/gopal.sharma/datasets/deep_text/text-non-text/neg_examples.hdf5', 'r') as hf:
    print('List of arrays in this file: \n', hf.keys())
    data = hf.get('X')
    X_neg = np.array(data)
    data = hf.get('y')
    y_neg = np.array(data, dtype = np.int8) - 1
print ((y_neg.shape, y_neg[0]))
X_neg = np.expand_dims(X_neg, axis = 1)
X = np.concatenate((X_pos, X_neg), axis = 0)
y = np.concatenate((y_pos, y_neg), axis = 0)

# Print the size of the inputs
print('Size of the X: ', X.shape)
print('Size of y: ', y.shape)

# Preprocess the data: zero centered and unit std
X = X.astype('float32')
X /= 255
X -= X.mean()
X /= X.std()

# Generate the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=100)

# Let us see the size
print('X_train size: ', X_train.shape)
print("X_test size: ", X_test.shape)

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

print('shape of y_train: ', y_train.shape)
print('shape of y_test: ', y_test.shape)

model = Sequential()
model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, img_rows, img_cols), trainable= False))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', trainable= False))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', trainable= False))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.75))

model.add(Convolution2D(256, 3, 3, trainable= False))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Convolution2D(256, 3, 3, trainable= False))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add((Dense(2, init='he_normal')))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(nb_classes, init='he_normal'))
model.add(Activation('softmax'))


print (model.summary())
transfer = Sequential()
tranfer = weight_loading(transfer)
# Since I know the keys in the weights.hdf5 file, I can directly take the weights.
layers = [0, 2, 4, 8, 11]

for l in layers:
    model.layers[l].set_weights(transfer.layers[l].get_weights())

print('Model loaded.')

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
print(model.summary())

checkpointer = ModelCheckpoint(filepath="weights_new.hdf5", verbose=1, save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
# # Object of the class history
history = LossHistory()

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, y_train,
              batch_size=batch_size,
              nb_epoch=1,
              validation_data=(X_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do pre-processing and realtime data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.10,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.10,  # randomly shift images vertically (fraction of total height)
        vertical_flip=False,
        zoom_range=0.1,
        shear_range=0.1,
        zca_whitening=True,
        fill_mode="nearest")  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, y_train,
                                     batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, y_test),
                        callbacks=[history, checkpointer, early_stopping])
    score = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
    print (score[0])
    print (score[1])
    f = open('out.txt', 'w')
    print >> f, 'Losses:', score[0]  # or f.write('...\n')
    print >> f, "Score:", score[1]
    f.close()
    # Extracting the losses and accuracies
    val_losses = history.val_losses_epoch
    val_axis = np.arange(len(val_losses))
    losses_batch = history.losses_batch
    loss_batch_axis = np.arange(len(losses_batch))
    acc = history.acc
    acc_axis = np.arange(len(acc))

    plt.subplot(1, 2, 1)
    plt.plot(val_axis, val_losses, color="blue", label="validation_loss")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.plot(loss_batch_axis, losses_batch, color="red", label="training_loss")
    plt.axis('off')
    plt.savefig('losses.png')

    plt.plot(acc_axis, acc, label="Accury v/s epochs")
    plt.axis('off')
    plt.savefig('accuracy.png')
with h5py.File('losses.h5', 'w') as hf:
    hf.create_dataset('acc', data=acc)
    hf.create_dataset('val_losses', data=val_losses)
    hf.create_dataset('losses_batch', data=losses_batch)