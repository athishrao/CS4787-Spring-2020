import os
import numpy as np
import scipy
import matplotlib
import datetime
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot as plt

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# additional imports you may find useful for this assignment
from tqdm import tqdm
from scipy.special import softmax

# TODO add any additional imports and global variables

def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training()
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = np.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        Xs_te, Lbls_te = mnist_data.load_testing()
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label

        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the cross-entropy loss of the classifier
#
# x         examples          (d)
# y         labels            (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    yHat = softmax(np.dot(W,x.T))
    yHat = np.log(yHat)
    ans = -1 * np.dot(y, yHat)
    ans += (gamma/2)*np.linalg.norm(W, 'fro')
    return ans

# compute the gradient of a single example of the multinomial logistic regression objective, with regularization
#
# x         training example   (d)
# y         training label     (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the model parameters
def multinomial_logreg_grad_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    yHat = softmax(np.dot(W,x.T))
    yHat -= y.T
    ans = np.dot(yHat, x)
    return ans


# test that the function multinomial_logreg_grad_i is indeed the gradient of multinomial_logreg_loss_i
def test_gradient(Xs, Ys, gamma, W):
    # TODO students should implement this in Part 1
    # d,c = W.shape
    # extend = [0 for _ in range((c-1)*d)]
    # print(Xs)
    # Xs = np.concatenate((Xs.T, extend), axis=None)
    # print(Xs.shape)
    # print(W.shape)
    pass


#
# compute the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should implement this
    pass

# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the model parameters
def multinomial_logreg_total_grad(Xs, Ys, gamma, W):
    # TODO students should implement this
    # a starter solution using an average of the example gradients
    ###################################
    # Starter Code
    # (d,n) = Xs.shape
    # acc = W * 0.0
    # for i in range(n):
    #     acc += multinomial_logreg_grad_i(Xs[:,i].T, Ys[:,i].T, gamma, W)
    # return acc / n

    ###################################
    # Numpy Code
    num_samples = Xs.shape[0]
    y_hat = softmax(np.dot(W,Xs.T), axis=0)
    del_L = np.dot(y_hat - Ys.T, Xs)/num_samples
    del_R = del_L + gamma*W
    return del_R
    ###################################

# compute the cross-entropy loss of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_total_loss(Xs, Ys, gamma, W):
    # TODO students should implement this
    # a starter solution using an average of the example gradients
    ###################################
    # Starter Code
    # (d,n) = Xs.shape
    # acc = W * 0.0
    # for i in range(n):
    #     acc += multinomial_logreg_loss_i(Xs[:,i].T, Ys[:,i].T, gamma, W)
    # return acc / n

    ###################################
    # Numpy Code
    num_examples = Xs.shape[0]
    y_hat = softmax(np.dot(W,Xs.T), axis=0)
    log_y_hat = -1 * np.log(y_hat)
    y_dot_y_hat = np.multiply(log_y_hat, Ys.T)
    L_y_y_hat = np.sum(y_dot_y_hat)/(num_examples)
    R = L_y_y_hat + (gamma/2)*(np.linalg.norm(W, 'fro'))**2
    return R
    ###################################

# run gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs            training examples (d * n)
# Ys            training labels   (d * c)
# gamma         L2 regularization constant
# W0            the initial value of the parameters (c * d)
# alpha         step size/learning rate
# num_iters     number of iterations to run
# monitor_freq  how frequently to output the parameter vector
#
# returns       a list of models parameters, one every "monitor_freq" iterations
#               should return model parameters before iteration 0, iteration monitor_freq, iteration 2*monitor_freq, and again at the end
#               for a total of (num_iters/monitor_freq)+1 models, if num_iters is divisible by monitor_freq.
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_iters, monitor_freq):
    # TODO students should implement this
    start_time = datetime.datetime.now()
    freq = []
    loss = []
    for i in range(num_iters):
        if (i % monitor_freq == 0):
            freq.append(W0)
            loss.append(multinomial_logreg_total_loss(Xs, Ys, gamma, W0))
        W0 -= alpha*multinomial_logreg_total_grad(Xs, Ys, gamma, W0)
    freq.append(W0)
    end_time = datetime.datetime.now()
    return freq, loss, end_time - start_time

# estimate the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
# nsamples  number of samples to use for the estimation
#
# returns   the estimated model error when sampling with replacement
def estimate_multinomial_logreg_error(Xs, Ys, W, nsamples):
    # TODO students should implement this
    pass

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # Convert all marices to np matrices
    Xs_tr, Xs_te, Ys_tr, Ys_te = np.matrix(Xs_tr.T), np.matrix(Xs_te.T), np.matrix(Ys_tr.T), np.matrix(Ys_te.T)
    print("Shape of initial X:", Xs_tr.shape)
    print("Shape of initial Y:", Ys_tr.shape)

    #Pass one example into function_i
    # x = Xs_tr[0]
    # y = Ys_tr[0]
    W = np.random.rand(Ys_tr.shape[1], Xs_tr.shape[1])

    # W = np.zeros([Ys_tr.shape[1], Xs_tr.shape[1]])
    # multinomial_logreg_loss_i(x, y, gamma, W)
    # multinomial_logreg_grad_i(x, y, gamma, W)
    # test_gradient(x, y, gamma, W)
    #Pass Entire Dataset

    # Part 2
    alpha = 1.0
    gamma = 0.0001
    numberIter = 100
    monitorFreq = 10
    print(f"Running numpy implementation config: alpha={alpha}, gamma={gamma}, #iterations={numberIter}, monitorFreq={monitorFreq}")
    W, loss, time_taken = gradient_descent(Xs_tr, Ys_tr, gamma, W, alpha, numberIter, monitorFreq)
    print(f"Time taken for the above config is:  {time_taken}")

    plt.plot(loss, range(1,numberIter//monitorFreq+1))
    plt.savefig("myGraphi_"+str(numberIter)+".png")
