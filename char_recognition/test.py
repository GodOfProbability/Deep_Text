'''
This is a classifier to be trained according to the first stage of the paper "Deep Features
 for Text Recognitions". The dataset includes the 36 classes provided by the website of the sa
 paper. Moreover, one more class is adeed as a background from 10 k examples from batch-1 of cifar-10
 More information is coming......
'''
# from __future__ import print_function
import matplotlib

matplotlib.use('Agg')
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, MaxoutDense
# from keras.optimizers import SGD
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
import h5py


batch_size = 512
nb_epoch = 50
data_augmentation = True
img_rows, img_cols = 24, 24
nb_classes = 37

with h5py.File('test_case_insesitive.h5','r') as hf:
    print('List of arrays in this file: \n', hf.keys())
    data = hf.get('X')
    X = np.array(data)
    data = hf.get('y')
    y = np.array(data)
X = np.expand_dims(X, axis = 1)

# Print the size of the inputs
print('Size of the X: ', X.shape)
print('Size of y: ', y.shape)

# Preprocess the data: zero centered and unit std
X = X.astype('float32')
X /= 255
X -= X.mean()
X /= X.std()

model = Sequential()
model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.75))

model.add(Convolution2D(256, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Convolution2D(256, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(37))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
print (model.summary())
model.load_weights('weights.hdf5')

score = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
print (score[0])
print (score[1])
f = open('out.txt', 'w')
print >> f, 'Losses:', score[0]  # or f.write('...\n')
print >> f, "Score:", score[1]
f.close()

