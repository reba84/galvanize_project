import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import os

def image_processing(train_dir, test_dir, img_width, img_height, batch_size):

    '''
    use once to vary images once cnn is working

    '''
    # keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    #     samplewise_center=False,
    #     featurewise_std_normalization=False,
    #     samplewise_std_normalization=False,
    #     zca_whitening=False,
    #     rotation_range=0.,
    #     width_shift_range=0.,
    #     height_shift_range=0.,
    #     shear_range=0.,
    #     zoom_range=0.,
    #     channel_shift_range=0.,
    #     fill_mode='nearest',
    #     cval=0.,
    #     horizontal_flip=False,
    #     vertical_flip=False,
    #     rescale=None,
    #     dim_ordering=K.image_dim_ordering())

    train_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)


    train_processing = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            color_mode = "grayscale"
            #class_mode='binary',
            shuffle=True)


    test_processing = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            color_mode = "grayscale"
            #class_mode='binary',
            shuffle=True)

    classes = train_processing.nb_class
    n_train_samples = train_processing.nb_sample
    n_test_samples = test_processing.nb_sample

    return train_processing, test_processing, classes, n_train_samples, n_test_samples

def build_net(classes, img_width, img_height, nb_fitlers, pool_size, kernel_size):

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            input_shape=(img_width, img_height))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model

def fit_net(model, train_processing, n_train_samples, n_test_samples, epoch):
    fit = model.fit_generator(train_processing,
                        samples_per_epoch = n_train_samples
                        nb_epoch = epochs,
                        validation_data = test_processing,
                        nb_val_samples = n_test_samples)
                        accuracy = 'acc: {}, loss: {}, val_acc: {}, val_loss: {}'.format(*fit.history.values())
    return accuracy

if __name__ == '__main__':
    #Set Parameters
    img_width, img_height = 150, 150
    train_dir = '/train'
    test_dir = '/test'
    epoch = 10
    batch_size = 128
    pool_size = (2, 2)
    kernel_size = (3, 3)
    nb_filters = 32

    #fit_image_generators, build CNN, train_network, save history
    train_processing, test_processing, classes, n_train_samples, n_test_samples = fit_image_generators(train_dir, test_dir, img_width, img_height, batch_size)
    model = build_net(classes, img_width, img_height, nb_fitlers, pool_size, kernel_size)
    hist = fit_net(model, train_processing, epoch, n_train_samples, n_val_samples, batch_size)
