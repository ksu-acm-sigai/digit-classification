import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    """
    Load MNIST data from tf.keras.datasets package
    :return: None
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    return (x_train, y_train), (x_test, y_test)


def build_model(input_shape):
    """
    Create a Convolutional Neural Network with Keras's sequential API
    :param input_shape: dimensions of input data
    :return model: CNN model to classify digits
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=[tf.keras.metrics.SparseCategoricalAccuracy()], patience=2
    )
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.003, momentum=0.9
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["sparse_categorical_accuracy"])

    return model


def plot_history(history):
    """
    Plot the learning curves of
    :param history: learning history of model - must use sparse_categorical_crossentropy
    :return: None
    """
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'],
             label='Training Error')
    plt.plot(hist['epoch'], hist['val_loss'],
             label='Validation Error')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist["sparse_categorical_accuracy"],
             label='Training Accuracy')
    plt.plot(hist['epoch'], hist["val_sparse_categorical_accuracy"],
             label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()


def plot_accuracy(model, hist, test_data):
    """
    Evaluate and plot training, validation, and test accuracy
    :param model: CNN model
    :param hist: learning history of model
    :param test_data: data used to evaluate model
    :return: None
    """
    train_acc = hist.history['sparse_categorical_accuracy'][-1]
    val_acc = hist.history['val_sparse_categorical_accuracy'][-1]

    x_test, y_test = test_data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print('  Training accuracy =', train_acc)
    print('Validation accuracy =', val_acc)
    print('      Test accuracy =', test_acc)
    print()

    plt.bar(
        [1, 2, 3],
        [train_acc, val_acc, test_acc],
        tick_label=['Train', 'Validation', 'Test'],
        color=['r', 'g', 'b']
    )
    plt.title('Mean Squared Error')


def main():
    """
    Creates and trains a new model from scratch
    """
    train_data, test_data = load_data()
    x_train, y_train = train_data

    model = build_model(x_train[0].shape)
    hist = model.fit(x=x_train, y=y_train, epochs=6, validation_split=0.2)

    plot_history(hist)
    plot_accuracy(model, hist, test_data)

    model.save()


if __name__ == '__main__':
    main()
