import matplotlib
matplotlib.use('Agg')
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

def weight_loading(model):

    model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=(1, 24, 24)))
    model.add(Activation('relu'))
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

    model.load_weights('/home/perceptron/weights.hdf5')

    return model


