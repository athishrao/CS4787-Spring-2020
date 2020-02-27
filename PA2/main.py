import os
import numpy as np
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot as plt
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
        Xs_tr = numpy.ascontiguousarray(Xs_tr)
        Ys_tr = numpy.ascontiguousarray(Ys_tr)
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


# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    # TODO students should implement this
    Xs = Xs.T
    Ys = Ys.T
    batchSize = len(ii)
    totalLoss = 0
    X_batch = Xs[ii]
    Y_batch = Ys[ii]
    yHat = (softmax(np.matmul(W, X_batch.T), axis = 0) - Y_batch.T)
    ans = np.matmul(yHat, X_batch) + gamma*W
    return ans/batchSize

# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a fraction of incorrect labels (a number between 0 and 1)
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

# ALGORITHM 1: run stochastic gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of iterations (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" iterations
def stochastic_gradient_descent(Xs, Ys, gamma, W, alpha, num_epochs, monitor_period):
    # TODO students should implement this
    T = num_epochs*Xs.shape[1]
    params = []
    for i in range(T):
        if (i % monitor_period == 0):
            params.append(W)
        index = np.random.randint(0,Xs.shape[1])
        W = W - alpha*(multinomial_logreg_grad_i(Xs, Ys, [index], gamma, W))
    params.append(W)
    return params

# ALGORITHM 2: run stochastic gradient descent with sequential sampling order
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of iterations (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" iterations
def sgd_sequential_scan(Xs, Ys, gamma, W, alpha, num_epochs, monitor_period):
    # TODO students should implement this
    params = []
    for i in range(num_epochs):
        for j in range(Xs.shape[1]):
            if (j % monitor_period == 0):
                params.append(W)
            W = W - alpha*(multinomial_logreg_grad_i(Xs, Ys, [j], gamma, W))
        params.append(W)
    return params

# ALGORITHM 3: run stochastic gradient descent with minibatching
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
def sgd_minibatch(Xs, Ys, gamma, W, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    params = []
    T = num_epochs*Xs.shape[1]/N
    for i in range(T):
        if (i % monitor_period == 0):
            params.append(W)
        ii = []
        for j in range(B):
            ii.append(np.random.randint(0,Xs.shape[1]))
        W = W - alpha*(multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W))
    params.append(W)
    return params

# ALGORITHM 4: run stochastic gradient descent with minibatching and sequential sampling order
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
def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    params = []
    for t in range(num_epochs):
        if (i % monitor_period == 0):
            params.append(W)
        ii = [(t*B + i) for i in range(B)]
        W = W - alpha*(multinomial_logreg_grad_i(Xs, Ys, ii , gamma, W))
    params.append(W)
    return params


if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    print("Shape of initial X:", Xs_tr.shape)
    print("Shape of initial Y:", Ys_tr.shape)
    DIVIDER = "#"*20
    gamma = 0.0001
    alpha = 0.001
    num_epochs = 10
    monitor_period = 1000
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])

    # params = stochastic_gradient_descent(Xs_tr, Ys_tr, gamma, W, alpha, num_epochs, monitor_period)
    error = []
    # error_np_te = 0

    params = pickle.load(open("W.pickle", "rb"))
    for w in params:
        error.append(multinomial_logreg_error(Xs_tr, Ys_tr, w))
    pickle.dump(params, open( "W.pickle", "wb" ) )

    plt.plot(range(600+1), error)
    plt.savefig("results/entire_ds_error_train_"+str(1)+".png")
    plt.close()
    # return
