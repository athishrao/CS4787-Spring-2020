#!/usr/bin/env python3
import os
import copy
import numpy
import scipy
import mnist
import time
import pickle
import matplotlib
import itertools as it
from numpy import random
# from tensorflow import keras
# from tensorflow.keras import datasets, layers, models

# matplotlib.use('agg')
from matplotlib import pyplot as plt


import tensorflow as tf
# from tensorflow.keras.layers import Dense

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
    inp_shape = (Xs.shape[1], Xs.shape[2], Xs.shape[3])
    Ys = Ys.astype("float64")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=inp_shape))
    model.add(tf.keras.layers.Dense(d1, activation="relu", name="dense_1"))
    model.add(tf.keras.layers.Dense(d2, activation="relu", name="dense_2"))
    model.add(tf.keras.layers.Dense(10, activation="softmax", name="predictions"))

    opt = tf.keras.optimizers.SGD(
        lr=alpha, momentum=beta, nesterov=False,
    )
    lossFunc =tf.keras.losses.sparse_categorical_crossentropy

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

    inp_shape = (Xs.shape[1], Xs.shape[2], Xs.shape[3])
    Ys = Ys.astype("float64")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=inp_shape))
    model.add(tf.keras.layers.Dense(d1, activation="relu", name="dense_1"))
    model.add(tf.keras.layers.Dense(d2, activation="relu", name="dense_2"))
    model.add(tf.keras.layers.Dense(10, activation="softmax", name="predictions"))

    opt = tf.keras.optimizers.Adam(
        lr=alpha,
        beta_1=rho1,
        beta_2=rho2,
        epsilon=1e-07,

    )
    lossFunc =tf.keras.losses.sparse_categorical_crossentropy

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

    inp_shape = (Xs.shape[1], Xs.shape[2], Xs.shape[3])
    Ys = Ys.astype("float64")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=inp_shape))
    model.add(tf.keras.layers.Dense(d1, activation="relu", name="dense_1"))
    model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=beta))
    model.add(tf.keras.layers.Dense(d2, activation="relu", name="dense_2"))
    model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=beta))
    model.add(tf.keras.layers.Dense(10, activation="softmax", name="predictions"))

    opt = tf.keras.optimizers.SGD(
        lr=alpha, momentum=beta, nesterov=False,
    )
    lossFunc =tf.keras.losses.sparse_categorical_crossentropy

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
def train_CNN_sgd(Xs, Ys, alpha, rho1, rho2, B, Epochs):
    # TODO students should implement this

    inp_shape = (Xs.shape[1], Xs.shape[2], Xs.shape[3])
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), activation="relu", input_shape=inp_shape
        )
    )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    opt = tf.keras.optimizers.Adam(
        lr=alpha,
        beta_1=rho1,
        beta_2=rho2,
        epsilon=1e-07,
    )
    lossFunc = tf.keras.losses.sparse_categorical_crossentropy

    model.compile(optimizer=opt, loss=lossFunc, metrics=["acc"])
    history = model.fit(
        Xs, Ys, batch_size=B, epochs=Epochs, validation_split=validation_percentage
    )

    return model, history


def run_algo(algorithm_identifier, algo_fn, algo_args, X_te, Y_te, names):
    print(f"\nRunning Algorithm {algorithm_identifier} ...")
    start = time.time()
    model, history = algo_fn(**algo_args)
    end = time.time()
    time_taken = end - start
    print(f"Algorithm {algorithm_identifier} complete. Time taken: {time_taken}")

    print(f"Testing Algorithm {algorithm_identifier} ...")
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
    plt.plot(
        range(len(history["val_loss"])), history["val_loss"], label="Validation loss"
    )
    plt.plot(range(len(history["loss"])), test_err, label="Testing loss")
    plt.title(names[algorithm_identifier])
    plt.xlabel("Number of Epoch")
    plt.ylabel("loss")
    plt.legend(loc="lower right")
    plt.savefig(figures_dir + algorithm_identifier + "_loss" + ".png")
    plt.close()

    test_acc = [test_acc for _ in range(len(history["loss"]))]
    plt.plot(range(len(history["acc"])), history["acc"], label="Training accuracy")
    plt.plot(
        range(len(history["val_acc"])), history["val_acc"], label="Validation accuracy"
    )
    plt.plot(range(len(history["val_acc"])), test_acc, label="Testing accuracy")
    plt.title(names[algorithm_identifier])
    plt.xlabel("Number of Epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="lower right")
    plt.savefig(figures_dir + algorithm_identifier + "_acc" + ".png")
    plt.close()

def print_config(config):
    print("Current Configuration:")
    print("~" * 15)
    for k in config:
        print(f"{k}: {config[k]}")
    print("~" * 15)

def grid_search(
    tr_data, te_data, hyper_params, algo_args, algo_fn, algo_id, names, tuning_basis=""
):
    train_x, train_y = tr_data
    test_x, test_y = te_data
    key_set = sorted(hyper_params)
    configs = [
        dict(zip(key_set, prod))
        for prod in it.product(*(hyper_params[k] for k in key_set))
    ]

    print("\n" + DIVIDER)
    print(DIVIDER)
    print(f"Performing hyperparameter tuning for {names[algo_id]} ...")
    for config in configs:
        print("\n")
        print_config(config)
        for k in config:
            algo_args[k] = config[k]
        model, history, te_l, te_acc, _ = run_algo(
            algo_id, algo_fn, algo_args, test_x, test_y, names
        )
        config["te_l"] = te_l
        config["te_acc"] = te_acc
        config["tr_loss"] = history.history["loss"][-1]
        config["val_loss"] = history.history["val_loss"][-1]
        print("\n")

    print(f"Tuning for {names[algo_id]} complete.")
    print(DIVIDER)
    print(DIVIDER)

    return choose_best(configs, metric=tuning_basis)


# Custom Argmax
# Choose best dict based on a certain key (metric)
# Find either max or min
def choose_best(dict_list, metric="", find_max=False):
    fn = min if not find_max else max
    if not metric:
        return fn(dict_list, key=lambda x: (x["val_loss"]))
    return fn(dict_list, key=lambda x: x[metric])


if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()

    # To make it run on not-so powerful machines
    # (Xs_tr, Ys_tr, Xs_te, Ys_te) = (Xs_tr[:50], Ys_tr[:50], Xs_te[:50], Ys_te[:50])

    DIVIDER = "#" * 20

    names = {
        "sgd_no_momen": "SGD with no Momentum",
        "sgd_momen": "SGD with Momentum",
        "adam": "SGD with Adam",
        "sgd_bn": "SGD with Batch Normalisation",
        "cnn": "CNN with Adams",
    }
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

    cnn_args = copy.copy(adam_args)
    del cnn_args["d1"]
    del cnn_args["d2"]

    algorithms = {
        "sgd_no_momen": (train_fully_connected_sgd, sgd_no_momen_args),
        "sgd_momen": (train_fully_connected_sgd, sgd_momen_args),
        "adam": (train_fully_connected_adam, adam_args),
        "sgd_bn": (train_fully_connected_bn_sgd, sgd_bn_args),
        "cnn": (train_CNN_sgd, cnn_args),
    }

    # Experiments
    for algorithm in algorithms:
        model, history, te_l, te_acc, time_taken = run_algo(
            algorithm,
            algorithms[algorithm][0],
            algorithms[algorithm][1],
            Xs_te,
            Ys_te,
            names,
        )
        print("History:", history.history)

    # Hyperparameter tuning for SGD with Momentum
    hyper_params = {"alpha" : [0.99, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]}

    sgd_momtm_tune = grid_search((Xs_tr, Ys_tr), (Xs_te, Ys_te), hyper_params, algorithms["sgd_momen"][1], algorithms["sgd_momen"][0], "sgd_momen", names)
    print("Best hyperparam combo for SGD with momentum is:")
    print(sgd_momtm_tune)

    # Hyperparameter tuning for SGD without Momentum
    hyper_params = {
        "alpha" : [0.3, 0.1, 0.01, 0.001],
        "d1" : [64, 256, 512],
        "d2" : [32, 128, 256],
    }
    sgd_tune = grid_search((Xs_tr, Ys_tr), (Xs_te, Ys_te), hyper_params, algorithms["sgd_no_momen"][1], algorithms["sgd_no_momen"][0], "sgd_no_momen", names)
    print("Best hyperparam combo for SGD without momentum is:")
    print(sgd_tune)

    r_hyperparams = {"alpha":[], "d1":[], "d2":[]}
    mu_d, sigma_d = 6, 2
    mu_a, sigma_a = 0.1, 0.02
    num_trials = 12
    for i in range(int(num_trials ** (1/3))):
        r_hyperparams["alpha"] += [sigma_a * numpy.random.randn() + mu_a]
        r_hyperparams["d1"] += [int(2 ** (sigma_d * numpy.random.randn() + mu_d))]
        r_hyperparams["d2"] += [int(2 ** (sigma_d * numpy.random.randn() + mu_d))]
    sgd_tune = grid_search((Xs_tr, Ys_tr), (Xs_te, Ys_te), r_hyperparams, algorithms["sgd_no_momen"][1], algorithms["sgd_no_momen"][0], "sgd_no_momen", names)
    print("Best hyperparam combo for SGD without momentum is:")
    print(sgd_tune)
