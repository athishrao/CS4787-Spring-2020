#!/usr/bin/env python3
import os
import numpy
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

from scipy.special import softmax

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        numpy.random.seed(8675309)
        perm = numpy.random.permutation(60000)
        Xs_tr = numpy.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = numpy.ascontiguousarray(Ys_tr[:,perm])
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
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
    yHat = softmax(np.dot(W,Xs),axis=0).T
    count = 0
    for i in range(len(Ys)):
        pred = np.argmax(yHat[i])
        if (Ys[i,pred] != 1):
            count += 1
    return count/len(Ys)

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
    yHat = softmax(np.dot(W,x))
    yHat = np.log(yHat)
    ans = -1 * np.dot(y.T, yHat)
    ans += (gamma/2)*(np.linalg.norm(W, 'fro'))**2
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
    loss = []
    error = []
    for i in range(num_iters):
        if (i % monitor_freq == 0):
            params.append(W0)
        W0 = W0 - alpha*multinomial_logreg_total_grad(Xs, Ys, gamma, W0, starter)
    params.append(W0)
    error.append(multinomial_logreg_error(Xs, Ys, W0))
    loss.append(multinomial_logreg_total_loss(Xs, Ys, gamma, W0, starter))
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
        if (i % monitor_freq == 0):
            params.append(W0)
        W0 = W0 - alpha*multinomial_logreg_total_grad(Xs, Ys, gamma, W0, starter)
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
def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
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
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, monitor_period):
    # TODO students should implement this
    params = []
    W = W0
    v = 0
    d,n = Xs.shape
    for t in range(0, num_epochs):
        for i in range(n//B-1):
            if i % monitor_period == 0:
                params.append(W)
            ii = [(j * B + i) for i in range(B)]
            g = (1/B)*(multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W))
            v = beta*v - alpha*g
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
    d,n = Xs.shape
    t = 0
    s = [0 for i in range(d)]
    r = [0 for i in range(d)]
    for k in range(0,num_epochs):
        for i in range(n//B-1):
            if i % monitor_period == 0:
                params.append(W)
            t += 1
            ii = [(j * B + i) for i in range(B)]
            g = (1/B)*(multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W))
            for j in range(d):
                s[j] = rho1*s[j] + (1-rho1)*g[j]
                r[j] = rho2*r[j] + (1-rho2)*g[j]**2
            s_cap = s/(1-(rho1**t))
            r_cap = r/(1-(rho2**t))
            for j in range(d):
                W0[j] = W0[j] - (alpha*s[j])/np.sqrt(r[j]+eps)
    params.append(W0)
    return params

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    pass
