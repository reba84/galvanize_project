import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import os
from datetime import datetime

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
            color_mode = "grayscale",
            #class_mode='binary',
            shuffle=True)


    test_processing = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            color_mode = "grayscale",
            #class_mode='binary',
            shuffle=True)

    classes = train_processing.nb_class
    n_train_samples = train_processing.nb_sample
    n_test_samples = test_processing.nb_sample

    return train_processing, test_processing, classes, n_train_samples, n_test_samples

def build_net(classes, img_width, img_height, nb_fitlers, pool_size, kernel_size):

    model = Sequential()

    model.add(Convolution2D(32, kernel_size[0], kernel_size[1], input_shape=(1, img_width, img_height)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    #model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(32, init = 'uniform'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    # model.add(Convolution2D(64, kernel_size[0], kernel_size[1]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.5))

    model.add(Dense(3, init = 'uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def fit_net(model, train_processing, test_processing, n_train_samples, n_test_samples, epoch):
    fit = model.fit_generator(train_processing, samples_per_epoch = n_train_samples, nb_epoch = epoch, validation_data = test_processing, nb_val_samples = n_test_samples)
    accuracy = 'acc: {}, loss: {}, val_acc: {}, val_loss: {}'.format(*fit.history.values())
    return fit


def save_model(fit, epoch, batch_size, classes, n_train_samples, n_test_samples):
    ## --- Save Settings ---
    datetime_str = str(datetime.now()).split('.')[0]

    #Save Weights & Model
    weights_path = 'weights/' + str(classes) + '.h5'
    architecture_path = 'weights/' + str(classes) + '.json'
    model.save_weights(weights_path, overwrite=True)
    model_json = model.to_json()
    with open(architecture_path, "w") as json_file:
        json_file.write(model_json)

    #Save Parameters and Accuracy
    parameters = '\nn_train_samples: {}, n_test_samples: {}, n_epoch: {}, batch_size: {}\n'.format(n_train_samples, n_test_samples, epoch, batch_size)
    accuracy = 'acc: {}, loss: {}, val_acc: {}, test_loss: {}'.format(*hist.history.values())
    text = '\n' + datetime_str + parameters + accuracy
    with open('log.txt', "a") as myfile:
        myfile.write(text)

    print "Saved!"

if __name__ == '__main__':
    #Set Parameters
    img_width, img_height = 150, 150
    train_dir = 'train'
    test_dir = 'test'
    epoch = 1
    batch_size = 128
    pool_size = (2, 2)
    kernel_size = (5, 5)
    nb_filters = 32

    #fit_image_generators, build CNN, train_network, save history
    train_processing, test_processing, classes, n_train_samples, n_test_samples = image_processing(train_dir, test_dir, img_width, img_height, batch_size)
    model = build_net(classes, img_width, img_height, nb_filters, pool_size, kernel_size)
    fit_model = fit_net(model, train_processing, test_processing, n_train_samples, n_test_samples, epoch)
    save_model(fit_model, epoch, batch_size, classes, n_train_samples, n_test_samples)
