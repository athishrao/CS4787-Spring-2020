#!/usr/bin/env python3
import os
import time
import scipy
import mnist
import pickle
import matplotlib
import numpy as np
import itertools as it
from numpy import random

matplotlib.use("agg")
from matplotlib import pyplot as plt
from scipy.special import softmax

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, "rb"))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training()
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = np.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        np.random.seed(8675309)
        perm = np.random.permutation(60000)
        Xs_tr = np.ascontiguousarray(Xs_tr[:, perm])
        Ys_tr = np.ascontiguousarray(Ys_tr[:, perm])
        Xs_te, Lbls_te = mnist_data.load_testing()
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, "wb"))
    return dataset


# compute the gradient of the multinomial logistic regression objective, with regularization (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W0         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W0):
    # TODO students should use their implementation from programming assignment 2
    d, n = Xs.shape
    c, n = Ys.shape
    batchSize = len(ii)
    X_batch = Xs[:, ii]
    Y_batch = Ys[:, ii]
    yHat = softmax(np.matmul(W0, X_batch), axis=0) - Y_batch
    ans = np.matmul(yHat, X_batch.T) + batchSize * gamma * W0
    return ans / batchSize


# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W0         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W0):
    # TODO students should use their implementation from programming assignment 1
    Ys = Ys.T
    yHat = softmax(np.dot(W0, Xs), axis=0).T
    count = 0
    for i in range(len(Ys)):
        pred = np.argmax(yHat[i])
        if Ys[i, pred] != 1:
            count += 1
    return count / len(Ys)


# compute the cross-entropy loss of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W0         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W0):
    # TODO students should implement this
    (d, n) = Xs.shape
    ret = 0
    # Numpy Code
    y_hat = softmax(np.dot(W0, Xs), axis=0)
    log_y_hat = -1 * np.log(y_hat)
    y_dot_y_hat = np.multiply(log_y_hat, Ys)
    L_y_y_hat = np.sum(y_dot_y_hat)
    ret = L_y_y_hat + (gamma / 2) * (np.linalg.norm(W0, "fro")) ** 2
    return ret / n


# gradient descent (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    # TODO students should use their implementation from programming assignment 1
    params = []
    grad_range = list(range(Xs.shape[1]))
    for i in range(num_epochs):
        if i % monitor_period == 0:
            params.append(W0)
        W0 = W0 - alpha * multinomial_logreg_grad_i(Xs, Ys, grad_range, gamma, W0)
    params.append(W0)
    return params


# gradient descent with nesterov momentum
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs
def multinomial_logreg_total_grad(Xs, Ys, gamma, W0):
    # TODO students should implement this
    # a starter solution using an average of the example gradients
    (d, n) = Xs.shape
    ret = 0
    # ----- NUMPY CODE
    y_hat = softmax(np.dot(W0, Xs), axis=0)
    del_L = np.dot(y_hat - Ys, Xs.T)
    ret = del_L + n * gamma * W0
    return ret / n


def gd_nesterov(Xs, Ys, gamma, W0, alpha, beta, num_epochs, monitor_period):
    # TODO students should implement this
    params = []
    loss = []
    error = []
    v = W0
    for i in range(num_epochs):
        if i % monitor_period == 0:
            params.append(W0)
        vPrev = v[:]
        v = W0 - alpha * multinomial_logreg_grad_i(
            Xs, Ys, range(Xs.shape[1]), gamma, W0
        )
        W0 = v + beta * (v - vPrev)
    params.append(W0)
    return params


# SGD: run stochastic gradient descent with minibatching and sequential sampling order (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_minibatch_sequential_scan(
    Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period
):
    # TODO students should use their implementation from programming assignment 2
    params = []
    for t in range(num_epochs):
        for j in range(Xs.shape[1] // B):
            if j % monitor_period == 0:
                params.append(W0)
            ii = [(j * B + i) for i in range(B)]
            W0 = W0 - alpha * (multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W0))
    params.append(W0)
    return params


# SGD + Momentum: add momentum to the previous algorithm
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_mss_with_momentum(
    Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, monitor_period
):
    # TODO students should implement this
    params = []
    W0 = W0
    v = 0
    d, n = Xs.shape
    for t in range(0, num_epochs):
        for i in range(n // B):
            if i % monitor_period == 0:
                params.append(W0)
            ii = [(i * B + j) for j in range(B)]
            g = multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W0)
            v = (beta * v) - (alpha * g)
            W0 = W0 + v
    params.append(W0)
    return params


# Adam Optimizer
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# rho1            first moment decay rate ρ1
# rho2            second moment decay rate ρ2
# B               minibatch size
# eps             small factor used to prevent division by zero in update step
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def adam(Xs, Ys, gamma, W0, alpha, rho1, rho2, B, eps, num_epochs, monitor_period):
    # TODO students should implement this
    params = []
    d, n = Xs.shape
    t = 0
    s = [0 for i in range(d)]
    r = [0 for i in range(d)]
    for k in range(0, num_epochs):
        for i in range(n // B):
            if i % monitor_period == 0:
                params.append(W0)
            t += 1
            ii = [(i * B + j) for j in range(B)]
            g = (multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W0))
            s = rho1 * np.asarray(s) + (1 - rho1) * np.asarray(g)
            r = rho2 * np.asarray(r) + (1 - rho2) * np.asarray(g) ** 2
            s_cap = np.array([i / (1 - (rho1 ** t)) for i in s])
            r_cap = np.array([i / (1 - (rho2 ** t)) for i in r])
            temp = alpha / np.sqrt(r_cap + eps)
            W0 = W0 - temp * s_cap
    params.append(W0)
    return params

def run_gd(
    Xs_inp,
    Ys_inp,
    pickle_file,
    algorithm_identifier,
    gd_fn,
    gd_args,
    ignore_pickle=False,
):
    print(f"\n{DIVIDER}\n")
    time_taken = 0
    if pickle_file and os.path.isfile(pickle_file) and not ignore_pickle:
        print(
            f"Algorithm {algorithm_identifier} weights exist at {pickle_file}. Algorithm - {algorithm_identifier} skipped."
        )
        W0 = pickle.load(open(pickle_file, "rb"))
        print(f"Algorithm {algorithm_identifier} params loaded")
    else:
        print(f"Running Algorithm {algorithm_identifier} ...")

        W0 = np.zeros([Ys_inp.shape[0], Xs_inp.shape[0]])
        start = time.time()

        gd_args["W0"] = W0
        gd_args["Xs"] = Xs_inp
        gd_args["Ys"] = Ys_inp
        W0 = gd_fn(**gd_args)

        end = time.time()
        time_taken = end - start
        print(f"Algorithm {algorithm_identifier} complete. Time taken: {time_taken}")
        if not ignore_pickle:
            print(f"Dumping params to {pickle_file} ...")
            pickle.dump(W0, open(pickle_file, "wb"))
            print(f"Dumping complete.")
    return W0, time_taken


def get_error(Xs, Ys, params):
    errors = []
    for w in params:
        errors.append(multinomial_logreg_error(Xs, Ys, w))
    return errors


def get_loss(Xs, Ys, params):
    losses = []
    for w in params:
        losses.append(multinomial_logreg_loss(Xs, Ys, gamma, w))
    return losses


def print_config(config):
    print("Current Configuration:")
    print("~" * 15)
    for k in config:
        print(f"{k}: {config[k]}")
    print("~" * 15)


# Custom GridSearch
# Returns only if tuning_basis has been provided (in which case it returns best config along with metrics)
# Otherwise returns None
def tune_hyperparams(
    tr_data, te_data, hyper_params, algo_args, algo_fn, algo_id, tuning_basis=""
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
    print(f"Performing hyperparameter tuning for {algo_id} ...")
    for config in configs:
        print_config(config)
        for k in config:
            algo_args[k] = config[k]
        params, _ = run_gd(train_x, train_y, "", algo_id, algo_fn, algo_args, True)
        tr_err, te_err = (
            min(get_error(train_x, train_y, params)),
            min(get_error(test_x, test_y, params)),
        )
        tr_loss = min(get_loss(train_x, train_y, params))
        print(f"Min. Training loss under current config is: {tr_loss}")
        print(f"Min. Training error under current config is: {tr_err}")
        print(f"Min. Testing error under current config is: {te_err}\n")
        config["tr_loss"] = tr_loss
        config["tr_err"] = tr_err
        config["te_err"] = te_err
    print(f"Tuning for {algo_id} complete.")
    print(DIVIDER)
    print(DIVIDER)

    return choose_best(tuning_basis, configs) if tuning_basis else None


# Custom Argmax
# Choose best dict based on a certain key (metric)
# Find either max or min
def choose_best(metric, dict_list, find_max=False):
    if not find_max:
        return min(dict_list, key=lambda x: x[metric])
    return max(dict_list, key=lambda x: x[metric])


def generatePlot(weight, lossOrError, name, questionNumber, color="green"):
    figures_dir = "Figures/"
    if not os.path.isdir(figures_dir):
        print("Figures folder does not exist. Creating ...")
        os.makedirs(figures_dir)
        print(f"Created {figures_dir}.")
    plt.plot(range(len(weight)), lossOrError)
    plt.xlabel("Number of Observations")
    plt.ylabel(name + "_error")
    plt.savefig(figures_dir + name + "_" + questionNumber + "_" + ".png")
    plt.close()


if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    pickle_file_dir = "pickle_files/"
    # Create pickle folder if not exists already
    if not os.path.isdir(pickle_file_dir):
        print("Pickle folder does not exist. Creating ...")
        os.makedirs(pickle_file_dir)
        print(f"Created {pickle_file_dir}.")
    DIVIDER = "#" * 20

    # --------------- PART 1 BEGINS ---------------
    gd_pickle_file = pickle_file_dir + "gd.pickle"
    nesterov_pickle_file = pickle_file_dir + "nesterov"
    gamma = 10 ** -4
    alpha = 1.0
    num_epochs = 100
    monitor_freq = 1

    # Part 1.6
    gd_args = {
        "Xs": Xs_tr,
        "Ys": Ys_tr,
        "W0": 0,
        "gamma": gamma,
        "alpha": alpha,
        "num_epochs": num_epochs,
        "monitor_period": monitor_freq,
    }

    w_gd, time_gd = run_gd(
        Xs_tr, Ys_tr, gd_pickle_file, "basic_gd", gradient_descent, gd_args
    )

    nesterov_gd_args = dict(gd_args)
    nesterov_gd_args["beta"] = 0.9
    w_nest_09, time_nest = run_gd(
        Xs_tr,
        Ys_tr,
        nesterov_pickle_file + "_beta_09.pickle",
        "nesterov_beta_09",
        gd_nesterov,
        nesterov_gd_args,
    )
    nesterov_gd_args["beta"] = 0.99
    w_nest_099, time_nest = run_gd(
        Xs_tr,
        Ys_tr,
        nesterov_pickle_file + "_beta_099.pickle",
        "nesterov_beta_099",
        gd_nesterov,
        nesterov_gd_args,
    )
    # Part 1.7
    gd_tr_err, gd_te_err = (
        get_error(Xs_tr, Ys_tr, w_gd),
        get_error(Xs_te, Ys_te, w_gd),
    )
    gd_tr_loss = get_loss(Xs_tr, Ys_tr, w_gd)

    nesterov_gd_09_tr_err, nesterov_gd_09_te_err = (
        get_error(Xs_tr, Ys_tr, w_nest_09),
        get_error(Xs_te, Ys_te, w_nest_09),
    )
    nesterov_gd_09_tr_loss = get_loss(Xs_tr, Ys_tr, w_nest_09)

    nesterov_gd_99_tr_err, nesterov_gd_99_te_err = (
        get_error(Xs_tr, Ys_tr, w_nest_099),
        get_error(Xs_te, Ys_te, w_nest_099),
    )
    nesterov_gd_099_tr_loss = get_loss(Xs_tr, Ys_tr, w_nest_099)

    # Part 1.8
    generatePlot(w_gd, gd_tr_err, "gd_tr_err", "1.8")
    generatePlot(w_gd, gd_te_err, "gd_te_err", "1.8")
    generatePlot(w_gd, gd_tr_loss, "gd_tr_loss", "1.8")

    generatePlot(w_nest_09, nesterov_gd_09_tr_err, "nesterov_gd_09_tr_err", "1.8")
    generatePlot(w_nest_09, nesterov_gd_09_te_err, "nesterov_gd_09_te_err", "1.8")
    generatePlot(w_nest_09, nesterov_gd_09_tr_loss, "nesterov_gd_09_tr_loss", "1.8")

    generatePlot(w_nest_099, nesterov_gd_99_tr_err, "nesterov_gd_99_tr_err", "1.8")
    generatePlot(w_nest_099, nesterov_gd_99_te_err, "nesterov_gd_99_te_err", "1.8")
    generatePlot(w_nest_099, nesterov_gd_099_tr_loss, "nesterov_gd_099_tr_loss", "1.8")

    # Part 1.9
    gd_time, nes_time = 0, 0
    for i in range(5):
        _, t_nest = run_gd(
            Xs_tr,
            Ys_tr,
            nesterov_pickle_file + "_beta_099",
            "nesterov_beta_099",
            gd_nesterov,
            nesterov_gd_args,
            True,
        )
        _, t_gd = run_gd(
            Xs_tr, Ys_tr, gd_pickle_file, "basic_gd", gradient_descent, gd_args, True
        )
        gd_time += t_gd / 5
        nes_time += t_nest / 5
    print(DIVIDER)
    print(f"Average time for Basic GD for 5 total runs is: {gd_time}")
    print(f"Average time for Nesterov GD for 5 total runs is: {nes_time}")

    # Part 1.10
    hyperpar = {"alpha": [0.25, 0.5, 0.75]}
    # ONLY PRINTS FOR NOW, RETURNS NOTHING BEC NO TUNING BASIS PASSED AS PARAM
    tune_hyperparams(
        (Xs_tr, Ys_tr), (Xs_te, Ys_te), hyperpar, gd_args, gradient_descent, "basic_gd"
    )
    hyperpar["beta"] = [0.5, 0.8, 0.925, 0.95]
    # ONLY PRINTS FOR NOW, RETURNS NOTHING BEC NO TUNING BASIS PASSED AS PARAM
    tune_hyperparams(
        (Xs_tr, Ys_tr),
        (Xs_te, Ys_te),
        hyperpar,
        nesterov_gd_args,
        gd_nesterov,
        "nesterov_gd",
    )

    # --------------- PART 2 BEGINS ---------------

    # Part 2.3
    sgd_pickle_file = pickle_file_dir + "sgd.pickle"
    alpha = 0.2
    B = 600
    num_epochs = 10
    monitor_period = 10

    sgd_args = dict(gd_args)
    sgd_args["alpha"] = 0.2
    sgd_args["num_epochs"] = 10
    sgd_args["monitor_period"] = 10
    sgd_args["B"] = 600

    w_sgd, time_sgd = run_gd(
        Xs_tr,
        Ys_tr,
        sgd_pickle_file,
        "basic_sgd",
        sgd_minibatch_sequential_scan,
        sgd_args,
    )

    momen_sgd_pickle_file = pickle_file_dir + "sgd_momentum"
    momentum_sgd_args = dict(sgd_args)

    momentum_sgd_args["beta"] = 0.9
    w_sgd_momen_09, time_sgd_momen = run_gd(
        Xs_tr,
        Ys_tr,
        momen_sgd_pickle_file + "_beta_09.pickle",
        "momen_sgd_beta_09",
        sgd_mss_with_momentum,
        momentum_sgd_args,
    )

    momentum_sgd_args["beta"] = 0.99
    w_sgd_momen_099, time_sgd_momen = run_gd(
        Xs_tr,
        Ys_tr,
        momen_sgd_pickle_file + "_beta_099.pickle",
        "momen_sgd_beta_099",
        sgd_mss_with_momentum,
        momentum_sgd_args,
    )

    # Part 2.4 (Athish)
    sgd_tr_err, sgd_te_err = (
        get_error(Xs_tr, Ys_tr, w_sgd),
        get_error(Xs_te, Ys_te, w_sgd),
    )
    sgd_tr_loss = get_loss(Xs_tr, Ys_tr, w_sgd)

    sgd_momen_09_tr_err, sgd_momen_09_te_err = (
        get_error(Xs_tr, Ys_tr, w_sgd_momen_09),
        get_error(Xs_te, Ys_te, w_sgd_momen_09),
    )
    sgd_momen_09_tr_loss = get_loss(Xs_tr, Ys_tr, w_sgd_momen_09)

    sgd_momen_99_tr_err, sgd_momen_99_te_err = (
        get_error(Xs_tr, Ys_tr, w_sgd_momen_099),
        get_error(Xs_te, Ys_te, w_sgd_momen_099),
    )
    sgd_momen_99_tr_loss = get_loss(Xs_tr, Ys_tr, w_sgd_momen_099)

    # Part 2.5 (Athish)
    generatePlot(w_sgd, sgd_tr_err, "sgd_tr_err", "2.5")
    generatePlot(w_sgd, sgd_te_err, "sgd_te_err", "2.5")
    generatePlot(w_sgd, sgd_tr_loss, "sgd_tr_loss", "2.5")

    generatePlot(w_sgd_momen_09, sgd_momen_09_tr_err, "sgd_momen_09_tr_err", "2.5")
    generatePlot(w_sgd_momen_09, sgd_momen_09_te_err, "sgd_momen_09_te_err", "2.5")
    generatePlot(w_sgd_momen_09, sgd_momen_09_tr_loss, "sgd_momen_09_tr_loss", "2.5")

    generatePlot(w_sgd_momen_099, sgd_momen_99_tr_err, "sgd_momen_99_tr_err", "2.5")
    generatePlot(w_sgd_momen_099, sgd_momen_99_te_err, "sgd_momen_99_te_err", "2.5")
    generatePlot(w_sgd_momen_099, sgd_momen_99_tr_loss, "sgd_momen_99_tr_loss", "2.5")
    # Part 2.6
    sgd_time, sgd_momen_time = 0, 0
    for i in range(5):
        _, t_momen = run_gd(
            Xs_tr,
            Ys_tr,
            momen_sgd_pickle_file + "_beta_099",
            "momen_sgd_beta_099",
            sgd_mss_with_momentum,
            momentum_sgd_args,
            True,
        )
        _, t_sgd = run_gd(
            Xs_tr,
            Ys_tr,
            sgd_pickle_file,
            "basic_sgd",
            sgd_minibatch_sequential_scan,
            sgd_args,
            True,
        )
        sgd_time += t_sgd / 5
        sgd_momen_time += t_momen / 5
    print(DIVIDER)
    print(f"Average time for Basic SGD for 5 total runs is: {sgd_time}")
    print(f"Average time for Momentum SGD for 5 total runs is: {sgd_momen_time}")

    # # Part 2.7 (Unassigned)
    hyperpar = {"alpha": [0.25, 0.5, 0.75]}
    # ONLY PRINTS FOR NOW, RETURNS NOTHING BEC NO TUNING BASIS PASSED AS PARAM
    tune_hyperparams(
        (Xs_tr, Ys_tr),
        (Xs_te, Ys_te),
        hyperpar,
        sgd_args,
        sgd_minibatch_sequential_scan,
        "basic_sgd",
    )
    hyperpar["beta"] = [0.5, 0.8, 0.925, 0.95]
    # ONLY PRINTS FOR NOW, RETURNS NOTHING BEC NO TUNING BASIS PASSED AS PARAM
    tune_hyperparams(
        (Xs_tr, Ys_tr),
        (Xs_te, Ys_te),
        hyperpar,
        momentum_sgd_args,
        sgd_mss_with_momentum,
        "momen_sgd",
    )

    # --------------- PART 3 BEGINS ---------------

    # Part 3.2

    # ADAM
    adam_sgd_pickle_file = pickle_file_dir + "sgd_adam.pickle"
    adam_sgd_args = dict(sgd_args)
    adam_sgd_args["eps"] = 10 ** -5
    adam_sgd_args["alpha"] = 0.01
    adam_sgd_args["rho1"] = 0.9
    adam_sgd_args["rho2"] = 0.999
    w_sgd_adam, time_adam = run_gd(
        Xs_tr, Ys_tr, adam_sgd_pickle_file, "adam_sgd", adam, adam_sgd_args
    )

    # Part 3.3
    sgd_tr_err, sgd_te_err = (
        get_error(Xs_tr, Ys_tr, w_sgd),
        get_error(Xs_te, Ys_te, w_sgd),
    )
    sgd_tr_loss = get_loss(Xs_tr, Ys_tr, w_sgd)

    sgd_adam_tr_err, sgd_adam_te_err = (
        get_error(Xs_tr, Ys_tr, w_sgd_adam),
        get_error(Xs_te, Ys_te, w_sgd_adam),
    )
    sgd_adam_tr_loss = get_loss(Xs_tr, Ys_tr, w_sgd_adam)

    # Part 3.4
    generatePlot(w_sgd, sgd_tr_err, "sgd_tr_err", "3.3")
    generatePlot(w_sgd, sgd_te_err, "sgd_te_err", "3.3")
    generatePlot(w_sgd, sgd_tr_loss, "sgd_tr_loss", "3.3")

    generatePlot(w_sgd_adam, sgd_adam_tr_err, "sgd_adam_tr_err", "3.3")
    generatePlot(w_sgd_adam, sgd_adam_te_err, "sgd_adam_te_err", "3.3")
    generatePlot(w_sgd_adam, sgd_adam_tr_loss, "sgd_adam_tr_loss", "3.3")

    # Part 3.5
    sgd_adam_time = 0
    for i in range(5):
        _, t_adam = run_gd(
            Xs_tr, Ys_tr, adam_sgd_pickle_file, "adam_sgd", adam, adam_sgd_args, True
        )
        sgd_adam_time += t_adam / 5
    print(DIVIDER)
    print(f"Average time for Basic SGD for 5 total runs is: {sgd_time}")
    print(f"Average time for Adam SGD for 5 total runs is: {sgd_adam_time}")

    # # Part 3.6
    hyperpar = {
        "alpha": [0.25, 0.5, 0.75],
        "rho1": [0.5, 0.8, 0.925, 0.95],
        "rho2": [0.5, 0.8, 0.925, 0.95],
    }
    # ONLY PRINTS FOR NOW, RETURNS NOTHING BEC NO TUNING BASIS PASSED AS PARAM
    tune_hyperparams(
        (Xs_tr, Ys_tr), (Xs_te, Ys_te), hyperpar, adam_sgd_args, adam, "adam_sgd"
    )
