import cv2
import numpy as np

from keras.datasets import cifar100
from keras import backend as K
from keras.utils import np_utils

#nb_train_samples = 3000 # 3000 training samples
#nb_valid_samples = 100 # 100 validation samples
num_classes = 100

def load_cifar100_data(img_rows, img_cols):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar100.load_data()
    nb_train_samples = len(X_train)
    nb_valid_samples = len(X_valid)

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_train, Y_train, X_valid, Y_valid

#if __name__=="__main__":
#	img_rows, img_cols = 224, 224 # Resolution of inputs
#	X_train, Y_train, X_valid, Y_valid = load_cifar100_data(img_rows, img_cols)
#	print ('X_train length: ', len(X_train))
#	print ('X_valid length: ', len(X_valid))
	