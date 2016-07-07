import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import h5py
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils

def data_utils():
    path_pos = '/data4/gopal.sharma/datasets/deep_text/text-non-text/train_text_non_text_pos.h5'
    path_neg = '/data4/gopal.sharma/datasets/deep_text/text-non-text/neg_examples.hdf5'
    path_test = '/data4/gopal.sharma/datasets/deep_text/char_recognition/test_case_insesitive.h5'

    with h5py.File(path_pos, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = hf.get('X')
        X_pos = np.array(data)
    plt.imshow(X_pos[0, :, :], cmap="Greys_r")
    plt.savefig('trial.png')

    X_pos = np.expand_dims(X_pos, axis=1)
    y_pos = np.ones((X_pos.shape[0], 1), dtype=np.uint8)
    print('Print the shape of y_pos: ', y_pos.shape, y_pos[0])

    with h5py.File(path_neg, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = hf.get('X')[0:200000]
        X_neg = np.array(data)

    X_neg = np.expand_dims(X_neg, axis=1)
    y_neg = np.zeros((X_neg.shape[0], 1), dtype=np.uint8)
    X = np.concatenate((X_pos, X_neg), axis=0)
    y = np.concatenate((y_pos, y_neg), axis=0)

    print(y_neg.shape, y_neg[0])
    print('Size of the X: ', X.shape)
    print('Size of y: ', y.shape)

    X = X.astype('float32')
    X /= 255
    X -= X.mean()
    X /= X.std()

    # Generate the split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)

    with h5py.File(path_test, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = hf.get('X')
        X = np.array(data)
        data = hf.get('y')
        y = np.array(data)
    X_test = np.expand_dims(X, axis=1)
    y_test = np.ones((X.shape[0], 1), dtype=np.uint8)

    # Let us see the size
    print('X_train size: ', X_train.shape)
    print("X_test size: ", X_test.shape)

    y_train = np_utils.to_categorical(y_train, 2)
    y_val = np_utils.to_categorical(y_val, 2)
    y_test = np_utils.to_categorical(y_test, 2)

    print('shape of y_train: ', y_train.shape)
    print('shape of y_test: ', y_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test