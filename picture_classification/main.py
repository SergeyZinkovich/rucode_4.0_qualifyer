from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import tensorflow as tf
import pandas as pd

import os
import numpy as np


def complete_model():
    inc_model=InceptionV3(include_top=False,
                          weights='imagenet',
                          input_shape=((250, 250, 3)))

    x = Flatten()(inc_model.output)
    x = Dense(256, name='dense_one')(x)
    # x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.5, name='dropout_one')(x)
    x = Dense(256, activation='relu', name='dense_two')(x)
    x = Dropout(0.5, name='dropout_two')(x)
    top_model=Dense(4, activation='sigmoid', name='output')(x)
    model = Model(input=inc_model.input, output=top_model)

    return model


if __name__ == "__main__":

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            'data2/train/',
            target_size=(250, 250),
            batch_size=32,
            class_mode='sparse')

    validation_generator = test_datagen.flow_from_directory(
            'data2/test/',
            target_size=(250, 250),
            batch_size=32,
            class_mode='sparse')

    pred_generator=test_datagen.flow_from_directory('data2/test/',
         target_size=(250,250),
         batch_size=32,
         class_mode='sparse')

    ttest_generator=test_datagen.flow_from_directory('data/test/', target_size=(250,250), batch_size=32, class_mode=None, shuffle=False)

    model=complete_model()

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        nb_epoch=10,
        validation_data=validation_generator,
        verbose=2)

    print('-'*50)
    print('Training the model has been completed')

    loss, accuracy = model.evaluate_generator(pred_generator, val_samples=100)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

    ans = model.predict_generator(ttest_generator)
    ans = np.array(ans)
    ans = ans.argmax(axis=1)

    ans = pd.DataFrame(ans)
    ans.to_csv("output.csv", index=False, header=False)

