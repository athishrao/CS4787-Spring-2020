import os
import numpy
import numpy as np
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)

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
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        Xs_te, Lbls_te = mnist_data.load_testing()
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
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
#####################ARR - CHECK########################################
# returns   the model cross-entropy loss
def multinomial_logreg_loss_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    W = np.matrix(W)
    # print(y.shape)
    expectedLoss = softmax(np.dot(W,x.T), axis=0)
    print(expectedLoss.shape, y.shape)
    y = y.T
    loss_i = -1 * sum([y[i]*np.log(expectedLoss[i]) for i in range(len(y))])
    return loss_i
#################################################################################

# compute the gradient of a single example of the multinomial logistic regression objective, with regularization
#
# x         training example   (d)
# y         training label     (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
#####################ARR - CHECK########################################
# returns   the gradient of the model parameters
def multinomial_logreg_grad_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    W = np.matrix(W)
    x = np.matrix(x)
    expectedLoss = softmax(np.dot(W,x.T), axis=0).T
    delF = np.dot((expectedLoss-y).T, x)
    print(delF.shape)
    return delF
#################################################################################

#####################ARR - CHECK########################################
# test that the function multinomial_logreg_grad_i is indeed the gradient of multinomial_logreg_loss_i
def test_gradient(x,y,gamma,W):
    # TODO students should implement this in Part 1
    alpha = 0.1**3
    # print("X ",x.shape)
    v = np.random.rand(1,x.shape[1])
    # print(v)
    new_x = x + alpha*v
    grad = (multinomial_logreg_loss_i(new_x,y,gamma,W) - multinomial_logreg_loss_i(x,y,gamma,W))/(alpha)
    # print(multinomial_logreg_grad_i(x, y, gamma, W).shape, v.shape)
    ###             MISTAKE
    ###             MISTAKE
    ###             MISTAKE
    multinomial_logreg_grad_i(x, y, gamma, W)
    grad_func = np.dot(v.T, grad)
    # print(grad_func, multinomial_logreg_grad_i(x, y, gamma, W))
    return grad_func == grad
#################################################################################

#####################ARR - CHECK########################################
# compute the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should implement this
    W = np.matrix(W)
    expected = softmax(np.dot(W,Xs), axis=0)
    error = sum([1 for i in range(len(Ys)) if (Ys[i]==expected[i])])/(len(Ys))
    return error
#################################################################################

#################################################################################
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
    
    # (d,n) = Xs.shape 
    # acc = W * 0.0
    # for i in range(n):
    #     acc += multinomial_logreg_grad_i(Xs[:,i], Ys[:,i], gamma, W)
    # return acc / n
    # print(Xs[0])
    return np.dot((softmax(np.dot(W, Xs.T), axis=0) - Ys.T),Xs) + gamma*W
#################################################################################

#################################################################################
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
    
    # (d,n) = Xs.shape
    # acc = W * 0.0
    # for i in range(n):
    #     acc += multinomial_logreg_loss_i(Xs[:,i], Ys[:,i], gamma, W)
    # return acc / n
    print(W[:,:5], "W")
    # print(Xs[:,:10], "X")
    print(np.dot(W, Xs.T)[:,:5], "aaaaaaaaaaaa")
    a = softmax(np.dot(W, Xs.T), axis=0)
    print(a[:,:5], "ssssssss")
    b = np.log(a)
    # print(b.shape, "b")
    # print(Ys.shape, "c")
    d = np.multiply(b.T, Ys)
    exit(0)
    # print(d.shape, "d")
    reg = np.linalg.norm(W, 'fro')
    return np.sum(d) + (gamma/2)*reg
    
#################################################################################

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
#################################################################################
# returns       a list of models parameters, one every "monitor_freq" iterations
#               should return model parameters before iteration 0, iteration monitor_freq, iteration 2*monitor_freq, and again at the end
#               for a total of (num_iters/monitor_freq)+1 models, if num_iters is divisible by monitor_freq.
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_iters, monitor_freq):
    # TODO students should implement this
    freq_monitor,loss = [], []
    # ADD TIME MEASUREMENT IMPLEMENTATION
    print("\nGradient Descent running ...\n")
    for i in range(num_iters):
        if i % monitor_freq == 0:
            freq_monitor += [W0]
        W0 = W0 - alpha * multinomial_logreg_total_grad(Xs, Ys, gamma, W0)
        loss += [multinomial_logreg_total_loss(Xs, Ys, gamma, W0)]
    freq_monitor += [W0]
    print("Gradient Descent complete.")
    return freq_monitor, loss

#################################################################################

# estimate the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
# nsamples  number of samples to use for the estimation
#
#################################################################################
# returns   the estimated model error when sampling with replacement
def estimate_multinomial_logreg_error(Xs, Ys, W, nsamples):
    # TODO students should implement this
    pass
#################################################################################

#################################################################################
if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    Xs_tr = np.matrix(Xs_tr).T
    Ys_tr = np.matrix(Ys_tr).T
    # print(Xs_tr)
    # print(Ys_tr)
    # W = np.random.rand(Ys_tr.shape[1],Xs_tr.shape[1])
    W = np.zeros([Ys_tr.shape[1],Xs_tr.shape[1]])
    test_gradient(Xs_tr[6], Ys_tr[6], 0.1, W)
    
    # plt.imshow(Xs_tr)s = 
    # plt.savefig("mygraph.png")
    gamma = 0.0001
    alpha = 1.0
    freq_params, loss = gradient_descent(Xs_tr, Ys_tr, gamma, W, alpha, 10, 10)
    # print(freq_params[1]-freq_params[0])
    
    fig = plt.figure()
    X = [i for i in range(1,11)]
    # Y = [2*i for i in range(1,11)]
    print(loss)
    plt.plot(X, loss)
    plt.savefig("mygraph.png")
    # TODO add code to produce figures

#################################################################################