#!/usr/bin/env python3
import os
import numpy as np
from numpy import random
import scipy
import matplotlib
import mnist
import time
import pickle

matplotlib.use("agg")
from matplotlib import pyplot

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
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    # TODO students should use their implementation from programming assignment 2
    Xs = Xs.T
    Ys = Ys.T
    batchSize = len(ii)
    totalLoss = 0
    X_batch = Xs[ii]
    Y_batch = Ys[ii]
    yHat = softmax(np.matmul(W, X_batch.T), axis=0) - Y_batch.T
    ans = np.matmul(yHat, X_batch) + gamma * W
    return ans / batchSize


# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should use their implementation from programming assignment 1
    Ys = Ys.T
    yHat = softmax(np.dot(W, Xs), axis=0).T
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
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):
    # TODO students should implement this
    yHat = softmax(np.dot(W, x))
    yHat = np.log(yHat)
    ans = -1 * np.dot(y.T, yHat)
    ans += (gamma / 2) * (np.linalg.norm(W, "fro")) ** 2
    ans = ans.item()
    return ans


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
def gd_nesterov(Xs, Ys, gamma, W0, alpha, beta, num_epochs, monitor_period):
    # TODO students should implement this
    params = []
    loss = []
    error = []
    for i in range(num_iters):
        if i % monitor_freq == 0:
            params.append(W0)
        W0 = W0 - alpha * multinomial_logreg_total_grad(Xs, Ys, gamma, W0, starter)
    params.append(W0)
    error.append(multinomial_logreg_error(Xs, Ys, W0))
    loss.append(multinomial_logreg_total_loss(Xs, Ys, gamma, W0, starter))
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
                params.append(W)
            ii = [(j * B + i) for i in range(B)]
            W = W - alpha * (multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W))
    params.append(W)
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
    W = W0
    v = 0
    d, n = Xs.shape
    for t in range(0, num_epochs):
        for i in range(n // B - 1):
            if i % monitor_period == 0:
                params.append(W)
            ii = [(j * B + i) for i in range(B)]
            g = (1 / B) * (multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W))
            v = beta * v - alpha * g
            W = W + v
    params.append(W)
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
        for i in range(n // B - 1):
            if i % monitor_period == 0:
                params.append(W)
            t += 1
            ii = [(j * B + i) for i in range(B)]
            g = (1 / B) * (multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W))
            for j in range(d):
                s[j] = rho1 * s[j] + (1 - rho1) * g[j]
                r[j] = rho2 * r[j] + (1 - rho2) * g[j] ** 2
            s_cap = s / (1 - (rho1 ** t))
            r_cap = r / (1 - (rho2 ** t))
            for j in range(d):
                W0[j] = W0[j] - (alpha * s[j]) / np.sqrt(r[j] + eps)
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
    time_taken = None
    if pickle_file and os.path.isfile(pickle_file) and not ignore_pickle:
        print(
            f"Algorithm {algorithm_identifier} weights exist at {pickle_file}. Algorithm - {algorithm_identifier} skipped."
        )
        W = pickle.load(open(pickle_file, "rb"))
        print(f"Algorithm {algorithm_identifier} params loaded")
    else:
        print(f"Running Algorithm {algorithm_identifier} ...")

        W = np.zeros([Ys_inp.shape[0], Xs_inp.shape[0]])
        start = time.time()

        gd_args["W0"] = W
        gd_args["Xs"] = Xs_inp
        gd_args["Ys"] = Ys_inp
        W = gd_fn(**gd_args)

        end = time.time()
        time_taken = end - start
        print(f"Algorithm {algorithm_identifier} complete. Time taken: {time_taken}")
        if not ignore_pickle:
            print(f"Dumping params to {pickle_file} ...")
            pickle.dump(W, open(pickle_file, "wb"))
            print(f"Dumping complete.")
    return W, time_taken

def get_error(Xs, Ys, params):
    errors = []
    for w in params:
        errors.append(multinomial_logreg_error(Xs, Ys, w))
    return errors

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    pickle_file_dir = "pickle_files/"
    DIVIDER = "#" * 20

    # --------------- PART 1 BEGINS ---------------
    gd_pickle_file = pickle_file_dir + "gd"
    nesterov_pickle_file = pickle_file_dir + "nesterov"
    gamma = 10 ** -4
    alpha = 1.0
    num_epochs = 100
    monitor_freq = 100

    # Part 1.6
    gd_args = {
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
        nesterov_pickle_file + "_beta_09",
        "nesterov_beta_09",
        gd_nesterov,
        nesterov_gd_args,
    )

    nesterov_gd_args["beta"] = 0.99
    w_nest_099, time_nest = run_gd(
        Xs_tr,
        Ys_tr,
        nesterov_pickle_file + "_beta_099",
        "nesterov_beta_099",
        gd_nesterov,
        nesterov_gd_args,
    )

    # TODO: Part 1.7 (Athish)
    gd_tr_err, gd_te_err = (get_error(X_tr, Y_tr, w_gd), get_error(X_tr, Y_tr, w_gd))

    nesterov_gd_09_tr_err, nesterov_gd_09_te_err = (get_error(X_tr, Y_tr, w_nest_09), get_error(X_tr, Y_tr, w_nest_09))

    nesterov_gd_99_tr_err, nesterov_gd_99_te_err = (get_error(X_tr, Y_tr, w_nest_99), get_error(X_tr, Y_tr, w_nest_99))

    # TODO: Part 1.8 (Athish)
    figures_dir = "Figures/"
    if not os.path.isdir(figures_dir):
        print("Figures folder does not exist. Creating ...")
        os.makedirs(figures_dir)
        print(f"Created {figures_dir}.")
    plt.plot(range(len(w_gd)), gd_tr_err)
    plt.savefig(figures_dir + "gd_tr" + "_1.8_" + ".png")
    plt.close()
    plt.plot(range(len(w_gd)), gd_te_err)
    plt.savefig(figures_dir + "gd_te" + "_1.8_" + ".png")
    plt.close()

    plt.plot(range(len(w_nest_09)), nesterov_gd_09_tr_err)
    plt.savefig(figures_dir + "nesterov_gd_09_tr_err" + "_1.8_" + ".png")
    plt.close()
    plt.plot(range(len(w_nest_09)), nesterov_gd_09_te_err)
    plt.savefig(figures_dir + "nesterov_gd_09_te_err" + "_1.8_" + ".png")
    plt.close()

    plt.plot(range(len(w_nest_99)), nesterov_gd_99_tr_err)
    plt.savefig(figures_dir + "nesterov_gd_99_tr_err" + "_1.8_" + ".png")
    plt.close()
    plt.plot(range(len(w_nest_99)), nesterov_gd_99_te_err)
    plt.savefig(figures_dir + "nesterov_gd_99_te_err" + "_1.8_" + ".png")
    plt.close()

    # Part 1.9
    gd_time, nes_time = time_gd / 5, time_nest / 5
    for i in range(4):
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

    # TODO Part 1.10 (Unassigned)

    # --------------- PART 2 BEGINS ---------------

    # Part 2.3
    sgd_pickle_file = pickle_file_dir + "sgd"
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
        momen_sgd_pickle_file + "_beta_09",
        "momen_sgd_beta_09",
        sgd_mss_with_momentum,
        momentum_sgd_args,
    )

    momentum_sgd_args["beta"] = 0.99
    w_sgd_momen_099, time_sgd_momen = run_gd(
        Xs_tr,
        Ys_tr,
        momen_sgd_pickle_file + "_beta_099",
        "momen_sgd_beta_099",
        sgd_mss_with_momentum,
        momentum_sgd_args,
    )

    # TODO: Part 2.4 (Athish)
    sgd_tr_err, sgd_te_err = (get_error(X_tr, Y_tr, w_sgd), get_error(X_tr, Y_tr, w_sgd))

    sgd_momen_09_tr_err, sgd_momen_09_te_err = (get_error(X_tr, Y_tr, w_sgd_momen_09), get_error(X_tr, Y_tr, w_sgd_momen_09))

    sgd_momen_99_tr_err, sgd_momen_99_te_err = (get_error(X_tr, Y_tr, w_sgd_momen_99), get_error(X_tr, Y_tr, w_sgd_momen_99))

    # TODO: Part 2.5 (Athish)
    plt.plot(range(len(w_sgd)), sgd_tr_err)
    plt.savefig(figures_dir + "sgd_tr_err" + "_2.5_" + ".png")
    plt.close()
    plt.plot(range(len(w_sgd)), sgd_te_err)
    plt.savefig(figures_dir + "sgd_te_err" + "_2.5_" + ".png")
    plt.close()

    plt.plot(range(len(w_sgd_momen_09)), sgd_momen_09_tr_err)
    plt.savefig(figures_dir + "sgd_momen_09_tr_err" + "_2.5_" + ".png")
    plt.close()
    plt.plot(range(len(w_sgd_momen_09)), sgd_momen_09_te_err)
    plt.savefig(figures_dir + "sgd_momen_09_te_err" + "_2.5_" + ".png")
    plt.close()

    plt.plot(range(len(w_sgd_momen_99)), sgd_momen_99_tr_err)
    plt.savefig(figures_dir + "sgd_momen_99_tr_err" + "_2.5_" + ".png")
    plt.close()
    plt.plot(range(len(w_sgd_momen_99)), sgd_momen_99_te_err)
    plt.savefig(figures_dir + "sgd_momen_99_te_err" + "_2.5_" + ".png")
    plt.close()

    # Part 2.6
    sgd_time, sgd_momen_time = time_sgd / 5, time_sgd_momen / 5
    for i in range(4):
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

    # TODO: Part 2.7 (Unassigned)

    # --------------- PART 3 BEGINS ---------------

    # Part 3.2

    # Use SGD results from above
    # ADAM
    adam_sgd_pickle_file = pickle_file_dir + "sgd_adam"
    adam_sgd_args = dict(sgd_args)
    adam_sgd_args["eps"] = 10 ** -5
    adam_sgd_args["alpha"] = 0.01
    adam_sgd_args["rho1"] = 0.9
    adam_sgd_args["rho2"] = 0.999
    w_sgd_adam, time_adam = run_gd(
        Xs_tr, Ys_tr, adam_sgd_pickle_file, "adam_sgd", adam, adam_sgd_args
    )

    # TODO: Part 3.3 (Athish)
    sgd_tr_err, sgd_te_err = (get_error(X_tr, Y_tr, w_sgd), get_error(X_tr, Y_tr, w_sgd))

    sgd_adam_tr_err, sgd_momen_09_te_err = (get_error(X_tr, Y_tr, w_sgd_adam), get_error(X_tr, Y_tr, w_sgd_adam))

    # TODO: Part 3.4 (Athish)
    plt.plot(range(len(w_sgd)), sgd_tr_err)
    plt.savefig(figures_dir + "sgd_tr_err" + "_3.3_" + ".png")
    plt.close()
    plt.plot(range(len(w_sgd)), sgd_te_err)
    plt.savefig(figures_dir + "sgd_te_err" + "_3.3_" + ".png")
    plt.close()

    plt.plot(range(len(w_sgd_adam)), sgd_adam_tr_err)
    plt.savefig(figures_dir + "sgd_adam_tr_err" + "_3.3_" + ".png")
    plt.close()
    plt.plot(range(len(w_sgd_adam)), sgd_adam_te_err)
    plt.savefig(figures_dir + "sgd_adam_te_err" + "_3.3_" + ".png")
    plt.close()

    # Part 3.5
    sgd_adam_time = time_adam / 5
    for i in range(4):
        _, t_adam = run_gd(
            Xs_tr, Ys_tr, adam_sgd_pickle_file, "adam_sgd", adam, adam_sgd_args, True
        )
        sgd_adam_time += t_adam / 5
    print(DIVIDER)
    print(f"Average time for Basic SGD for 5 total runs is: {sgd_time}")
    print(f"Average time for Adam SGD for 5 total runs is: {sgd_adam_time}")

    # TODO: Part 3.6 (Unassigned)
