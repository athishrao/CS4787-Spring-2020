#!/usr/bin/env python3
import os
import copy
import numpy
import scipy
import mnist
import time
import pickle
import matplotlib
from numpy import random
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

# matplotlib.use('agg')
from matplotlib import pyplot as plt


import tensorflow as tf
from tensorflow.keras.layers import Dense

mnist = tf.keras.datasets.mnist

if tf.__version__ != "2.1.0":
    print("-" * 40)
    print("**** warning: this assignment is tested on TensorFlow 2.1.0 or newer.")
    print(
        "****         if you run into issues, please check that you are updated to the latest stable TensorFlow version"
    )
    print("-" * 40)

# Constants
validation_percentage = 0.1

# load the MNIST dataset using TensorFlow/Keras
def load_MNIST_dataset():
    mnist = tf.keras.datasets.mnist
    (Xs_tr, Ys_tr), (Xs_te, Ys_te) = mnist.load_data()
    # plt.gray()
    # plt.imshow(Xs_tr[0])
    # plt.show()
    Xs_tr = Xs_tr / 255.0
    Xs_te = Xs_te / 255.0
    Xs_tr = Xs_tr.reshape(Xs_tr.shape[0], 28, 28, 1)  # 28 rows, 28 columns, 1 channel
    Xs_te = Xs_te.reshape(Xs_te.shape[0], 28, 28, 1)
    return (Xs_tr, Ys_tr, Xs_te, Ys_te)


# evaluate a trained model on MNIST data, and print the usual output from TF
#
# Xs        examples to evaluate on
# Ys        labels to evaluate on
# model     trained model
#
# returns   tuple of (loss, accuracy)
def evaluate_model(Xs, Ys, model):
    (loss, accuracy) = model.evaluate(Xs, Ys)
    return (loss, accuracy)


# train a fully connected two-hidden-layer neural network on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# beta      momentum parameter (0.0 if no momentum)
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_sgd(Xs, Ys, d1, d2, alpha, beta, B, Epochs):
    # TODO students should implement this

    Xs = Xs.reshape(60000, 784).astype("float64")
    Ys = Ys.astype("float64")

    model = models.Sequential()
    model.add(layers.Dense(d1, activation="relu", name="dense_1"))
    model.add(layers.Dense(d2, activation="relu", name="dense_2"))
    model.add(layers.Dense(10, activation="softmax", name="predictions"))

    opt = tf.keras.optimizers.SGD(
        learning_rate=alpha, momentum=beta, nesterov=False, name="SGD"
    )
    lossFunc = keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=opt, loss=lossFunc, metrics=["acc"])
    history = model.fit(
        Xs, Ys, batch_size=B, epochs=Epochs, validation_split=validation_percentage
    )

    return model, history


# train a fully connected two-hidden-layer neural network on MNIST data using Adam, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# rho1      first moment decay parameter
# rho2      second moment decay parameter
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_adam(Xs, Ys, d1, d2, alpha, rho1, rho2, B, Epochs):
    # TODO students should implement this

    Xs = Xs.reshape(60000, 784).astype("float64")
    Ys = Ys.astype("float64")

    model = models.Sequential()
    model.add(layers.Dense(d1, activation="relu", name="dense_1"))
    model.add(layers.Dense(d2, activation="relu", name="dense_2"))
    model.add(layers.Dense(10, activation="softmax", name="predictions"))

    opt = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=rho1,
        beta_2=rho2,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )
    lossFunc = keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=opt, loss=lossFunc, metrics=["acc"])
    history = model.fit(
        Xs, Ys, batch_size=B, epochs=Epochs, validation_split=validation_percentage
    )

    return model, history


# train a fully connected two-hidden-layer neural network with Batch Normalization on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# d1        the size of the first layer
# d2        the size of the second layer
# alpha     step size parameter
# beta      momentum parameter (0.0 if no momentum)
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_fully_connected_bn_sgd(Xs, Ys, d1, d2, alpha, beta, B, Epochs):
    # TODO students should implement this

    Xs = Xs.reshape(60000, 784).astype("float64")
    Ys = Ys.astype("float64")

    model = models.Sequential()
    model.add(layers.Dense(d1, activation="relu", name="dense_1"))
    model.add(layers.BatchNormalization(axis=-1, momentum=beta))
    model.add(layers.Dense(d2, activation="relu", name="dense_2"))
    model.add(layers.BatchNormalization(axis=-1, momentum=beta))
    model.add(layers.Dense(10, activation="softmax", name="predictions"))

    opt = tf.keras.optimizers.SGD(
        learning_rate=alpha, momentum=beta, nesterov=False, name="SGD"
    )
    lossFunc = keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=opt, loss=lossFunc, metrics=["acc"])
    history = model.fit(
        Xs, Ys, batch_size=B, epochs=Epochs, validation_split=validation_percentage
    )

    return model, history


# train a convolutional neural network on MNIST data using SGD, and print the usual output from TF
#
# Xs        training examples
# Ys        training labels
# alpha     step size parameter
# rho1      first moment decay parameter
# rho2      second moment decay parameter
# B         minibatch size
# epochs    number of epochs to run
#
# returns   a tuple of
#   model       the trained model (should be of type tensorflow.python.keras.engine.sequential.Sequential)
#   history     the history of training returned by model.fit (should be of type tensorflow.python.keras.callbacks.History)
def train_CNN_sgd(Xs, Ys, alpha, rho1, rho2, B, Epochs, d1, d2):
    # TODO students should implement this
    print(Xs.shape)
    input_shape = (28, 28, 1)
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            32, kernel_size=(3, 3), activation="relu", input_shape=input_shape
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    opt = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=rho1,
        beta_2=rho2,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )
    lossFunc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=opt, loss=lossFunc, metrics=["acc"])
    history = model.fit(
        Xs, Ys, batch_size=B, epochs=Epochs, validation_split=validation_percentage
    )

    return model, history


def run_algo(algorithm_identifier, algo_fn, algo_args, X_te, Y_te, names):
    print(f"Running Algorithm {algorithm_identifier} ...")
    start = time.time()
    model, history = algo_fn(**algo_args)
    end = time.time()
    time_taken = end - start
    print(f"Algorithm {algorithm_identifier} complete. Time taken: {time_taken}")

    print(f"Testing Algorithm {algorithm_identifier} ...")
    X_te = X_te.reshape(10000,784).astype("float64") if not (algorithm_identifier == "cnn") else X_te
    te_l, te_acc = evaluate_model(X_te, Y_te, model)
    print(f"Testing {algorithm_identifier} complete.")
    generatePlot(algorithm_identifier, history.history, te_l, te_acc, names)
    return model, history, te_l, te_acc, time_taken

def generatePlot(algorithm_identifier, history, test_err, test_acc, names):
    figures_dir = "Figures/"
    if not os.path.isdir(figures_dir):
        print("Figures folder does not exist. Creating ...")
        os.makedirs(figures_dir)
        print(f"Created {figures_dir}.")
    print(history, history["loss"], len(history["loss"]))
    test_err = [test_err for _ in range(len(history["loss"]))]
    plt.plot(range(len(history["loss"])), history["loss"], label="Training loss")
    plt.plot(range(len(history["val_loss"])), history["val_loss"], label="Training loss")
    plt.plot(range(len(history["loss"])), test_err, label="Training loss")
    plt.title(names[algorithm_identifier])
    plt.xlabel("Number of Epoch")
    plt.ylabel("loss")
    plt.legend(loc="lower right")
    plt.savefig(figures_dir + algorithm_identifier + "_loss" + ".png")
    plt.close()

    test_acc = [test_acc for _ in range(len(history["loss"]))]
    plt.plot(range(len(history["acc"])), history["acc"], label="Training accuracy")
    plt.plot(range(len(history["val_acc"])), history["val_acc"], label="Validation accuracy")
    plt.plot(range(len(history["val_acc"])), test_acc, label="Validation accuracy")
    plt.title(names[algorithm_identifier])
    plt.xlabel("Number of Epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.savefig(figures_dir + algorithm_identifier + "_acc" + ".png")
    plt.close()




if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()

    DIVIDER = "#" * 20
    basic_args = {
        "Xs": Xs_tr,
        "Ys": Ys_tr,
        "d1": 1024,
        "d2": 256,
        "alpha": 0.1,
        "Epochs": 10,
        "B": 128,
    }
    sgd_no_momen_args = copy.copy(basic_args)
    sgd_no_momen_args["beta"] = 0.0

    sgd_momen_args = copy.copy(sgd_no_momen_args)
    sgd_momen_args["beta"] = 0.9

    adam_args = copy.copy(basic_args)
    adam_args["alpha"] = 0.001
    adam_args["rho1"] = 0.99
    adam_args["rho2"] = 0.999

    sgd_bn_args = copy.copy(basic_args)
    sgd_bn_args["alpha"] = 0.001
    sgd_bn_args["beta"] = 0.9
    names = {
        "sgd_no_momen" : "SGD with no Momentum",
        "sgd_momen": "SGD with Momentum",
        "adam": "SGD with Adam",
        "sgd_bn": "SGD with Batch Normalisation",
        "cnn": "CNN with Adams",
    }
    algorithms = {
        "sgd_no_momen": (train_fully_connected_sgd, sgd_no_momen_args),
        "sgd_momen": (train_fully_connected_sgd, sgd_momen_args),
        "adam": (train_fully_connected_adam, adam_args),
        "sgd_bn": (train_fully_connected_bn_sgd, sgd_bn_args),
        "cnn": (train_CNN_sgd, adam_args),
    }

    # Experiments
    for algorithm in algorithms:
        model, history, te_l, te_acc, time_taken = run_algo(
            algorithm, algorithms[algorithm][0], algorithms[algorithm][1], Xs_te, Ys_te, names
        )
        # Generate Plots & Report here: Take a dict and unpack dict in plotting
        # te_l and te_acc are scalars, create an arr of len = len(x-axis marks) where each elem in the arr is the same value
        print(f"Time taken to run {algorithm}: {time_taken}")
        print("History:", history.history)
