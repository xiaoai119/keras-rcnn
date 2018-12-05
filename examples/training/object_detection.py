# -*- coding: utf-8 -*-

"""
Object detection
================

A simple example for ploting two figures of a exponential
function in order to test the autonomy of the gallery
stacking multiple images.
"""

import keras
import numpy

import keras_rcnn.datasets.shape
import keras_rcnn.models
import keras_rcnn.preprocessing
import keras_rcnn.utils
import matplotlib.pyplot as plt


def main():
    def draw_result():
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Loss')
        plt.legend()
        plt.show()

    training_dictionary, test_dictionary = keras_rcnn.datasets.shape.load_data()

    categories = {"circle": 1, "rectangle": 2, "triangle": 3}

    generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

    generator = generator.flow_from_dictionary(
        dictionary=training_dictionary,
        categories=categories,
        target_size=(224, 224)
    )

    validation_data = keras_rcnn.preprocessing.ObjectDetectionGenerator()

    validation_data = validation_data.flow_from_dictionary(
        dictionary=test_dictionary,
        categories=categories,
        target_size=(224, 224)
    )

    keras.backend.set_learning_phase(1)

    model = keras_rcnn.models.RCNN(
        categories=["circle", "rectangle", "triangle"],
        dense_units=512,
        input_shape=(224, 224, 3)
    )

    # optimizer = keras.optimizers.Adam()
    target = validation_data.next()[0][2]

    model.compile(loss='softmax', optimizer=keras.optimizers.RMSprop(lr=1e-4), metrics=['acc'])

    history = model.fit_generator(epochs=1, generator=generator, validation_data=validation_data, steps_per_epoch=100)
    x, y = model.predict(target)
    # model.save('rcnn.h5', overwrite=True)

    x = x[0]
    y = y[0]
    target = target[0]

    x = numpy.squeeze(x)
    y = numpy.squeeze(y)
    target = numpy.squeeze(target)
    numpy.amax(y, -1)
    keras_rcnn.utils.show_bounding_boxes(target, x, y)


if __name__ == '__main__':
    main()
