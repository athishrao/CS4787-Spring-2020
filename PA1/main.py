import os
import numpy as np
import scipy
import matplotlib
import datetime
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot as plt
from numpy.random import permutation
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
    yHat = softmax(np.dot(W,x))
    yHat = np.log(yHat)
    ans = -1 * np.dot(y.T, yHat)
    ans += (gamma/2)*(np.linalg.norm(W, 'fro'))**2
    ans = ans.item()
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
    yHat = softmax(np.matmul(W,x)) - y
    ans = np.matmul(yHat.reshape(-1,1), x.reshape(1,-1)) + gamma*W
    return ans

# test that the function multinomial_logreg_grad_i is indeed the gradient of multinomial_logreg_loss_i
def test_gradient(Xs, Ys, gamma, W, alpha):
    # TODO students should implement this in Part 1
    # ----- EXAMPLE CASE BEGINS
    # Xs = np.matrix([[-1], [0], [1]])
    # Ys = np.matrix([[1], [0]])
    # W = np.matrix([[1,2,3], [4,5,6]])
    # V = np.matrix([[1, 0, 0], [1, 0, 1]])
    # ----- EXAMPLE CASE ENDS

    num_examples = Xs.shape[1]
    count = 0
    for i in range(num_examples):
        X_i, Y_i = Xs[:,i], Ys[:,i]
        V = np.random.rand(W.shape[0], W.shape[1])
        # RHS
        func1 = multinomial_logreg_loss_i(X_i, Y_i, gamma, W + alpha*V)
        func2 = multinomial_logreg_loss_i(X_i, Y_i, gamma, W)
        RHS = (func1-func2)/alpha
        # LHS
        V = V.reshape(-1,1)
        grad = multinomial_logreg_grad_i(X_i, Y_i, gamma, W)
        grad = grad.reshape(-1,1)
        LHS = np.dot(V.T, grad).item()
        # Difference
        count += abs(LHS - RHS)
    return count / num_examples


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
    Ys = Ys.T
    yHat = softmax(np.dot(W,Xs),).T
    count = 0
    for i in range(len(Ys)):
        pred = np.argmax(yHat[i])
        if (Ys[i,pred] != 1):
            count += 1
    return count/len(Ys)

# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the model parameters
def multinomial_logreg_total_grad(Xs, Ys, gamma, W, starter=False):
    # TODO students should implement this
    # a starter solution using an average of the example gradients
    (d,n) = Xs.shape
    ret = 0
    if starter == True:
        # ----- STARTER CODE
        ret = W * 0.0
        for i in range(n):
            ret += multinomial_logreg_grad_i(Xs[:,i], Ys[:,i], gamma, W)
    else:
        # ----- NUMPY CODE
        y_hat = softmax(np.dot(W,Xs), axis=0)
        del_L = np.dot(y_hat - Ys, Xs.T)
        ret = del_L + gamma*W
    return ret / n

# compute the cross-entropy loss of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_total_loss(Xs, Ys, gamma, W, starter=False):
    # TODO students should implement this
    # a starter solution using an average of the example gradients
    (d,n) = Xs.shape
    ret = 0
    if starter == True:
        ###################################
        # Starter Code
        for i in range(n):
            ret += multinomial_logreg_loss_i(Xs[:,i], Ys[:,i], gamma, W)
    else:
        ###################################
        # Numpy Code
        y_hat = softmax(np.dot(W,Xs), axis=0)
        log_y_hat = -1 * np.log(y_hat)
        y_dot_y_hat = np.multiply(log_y_hat, Ys)
        L_y_y_hat = np.sum(y_dot_y_hat)
        ret = L_y_y_hat + (gamma/2)*(np.linalg.norm(W, 'fro'))**2
    return ret / n
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
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_iters, monitor_freq, starter=False):
    # TODO students should implement this
    params = []
    loss = []
    error = []
    for i in range(num_iters):
        if (i % monitor_freq == 0):
            params.append(W0)
            # error.append(multinomial_logreg_error(Xs, Ys, W0))
            # loss.append(multinomial_logreg_total_loss(Xs, Ys, gamma, W0, starter))
        # print(alpha*multinomial_logreg_total_grad(Xs, Ys, gamma, W0, starter))
        W0 = W0 - alpha*multinomial_logreg_total_grad(Xs, Ys, gamma, W0, starter)
        # if ((W0 - params[-1]).all() == 0):
        #     print("same")
    params.append(W0)
    error.append(multinomial_logreg_error(Xs, Ys, W0))
    loss.append(multinomial_logreg_total_loss(Xs, Ys, gamma, W0, starter))
    return params

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
    # X_sub, Y_sub = Xs.T, Ys.T
    perm = permutation(Xs.shape[1])
    X_sub = Xs.T[perm]
    Y_sub = Ys.T[perm]
    X_sub, Y_sub = X_sub[:nsamples], Y_sub[:nsamples]
    estimated_error = multinomial_logreg_error(X_sub.T, Y_sub.T, W)
    return estimated_error

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # Convert all marices to np matrices
    Xs_tr, Xs_te, Ys_tr, Ys_te = np.matrix(Xs_tr), np.matrix(Xs_te), np.matrix(Ys_tr), np.matrix(Ys_te)
    print("Shape of initial X:", Xs_tr.shape)
    print("Shape of initial Y:", Ys_tr.shape)

    DIVIDER = "#"*20

    # Part 1
    print(f"{DIVIDER}\nRunning part 1 ...\n")
    gamma = 0.0001
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
    alpha = [10**-1, 10**-3, 10**-5]
    for a in alpha:
        ret = test_gradient(Xs_tr, Ys_tr, gamma, W, a)
        print(f"For alpha={a}, average difference is: {ret}")
    print("\nPart 1 complete .\n")


    # Part 2
    print(f"{DIVIDER}\nRunning part 2 ...\n")
    gamma = 0.0001
    alpha = 1.0
    numberIter = 10
    monitorFreq = 10
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])

    print(f"Running starter code implementation config: alpha={alpha}, gamma={gamma}, #iterations={numberIter}, monitorFreq={monitorFreq}")
    start = datetime.datetime.now()
    Ws_starter = gradient_descent(Xs_tr, Ys_tr, gamma, W, alpha, numberIter, monitorFreq, True)
    end = datetime.datetime.now()
    print(f"Time taken for the above config is:  {end-start}")

    # Part 3
    print(f"{DIVIDER}\nRunning part 3 ...\n")
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
    end = datetime.datetime.now()

    print(f"Running numpy implementation config: alpha={alpha}, gamma={gamma}, #iterations={numberIter}, monitorFreq={monitorFreq}")
    start = datetime.datetime.now()
    Ws_numpy = gradient_descent(Xs_tr, Ys_tr, gamma, W, alpha, numberIter, monitorFreq)
    end = datetime.datetime.now()
    print(f"Time taken for the above config is:  {end-start}")
    print("\nPart 3 complete.\n")

    # Part 4
    print(f"{DIVIDER}\nRunning part 4 ...\n")
    numberIter = 1000
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
    print(f"Running numpy implementation config: alpha={alpha}, gamma={gamma}, #iterations={numberIter}, monitorFreq={monitorFreq}")
    start = datetime.datetime.now()
    Ws_numpy = gradient_descent(Xs_tr, Ys_tr, gamma, W, alpha, numberIter, monitorFreq)
    end = datetime.datetime.now()
    print(f"Time taken for the above config is:  {end-start}")

    est_err_tr_100, est_err_tr_1000, error, loss, est_err_te_1000, est_err_te_100, loss_np_te, error_np_te = [], [], [], [], [], [], [], []

    start = datetime.datetime.now()
    _ = multinomial_logreg_error(Xs_tr, Ys_tr, Ws_numpy[-1])
    end = datetime.datetime.now()
    print("\nRunning time to obtain training error with entire dataset on a single model is:", end - start)


    for w in Ws_numpy:

        loss.append(multinomial_logreg_total_loss(Xs_tr, Ys_tr, gamma, w))
        loss_np_te += [multinomial_logreg_total_loss(Xs_te, Ys_te, gamma, w)]

        error.append(multinomial_logreg_error(Xs_tr, Ys_tr, w))
        error_np_te += [multinomial_logreg_error(Xs_te, Ys_te, w)]

        num_ex = 100
        est_err_tr_100.append(estimate_multinomial_logreg_error(Xs_tr, Ys_tr, w, num_ex))
        est_err_te_100.append(estimate_multinomial_logreg_error(Xs_te, Ys_te, w, num_ex))

        num_ex = 1000
        est_err_tr_1000.append(estimate_multinomial_logreg_error(Xs_tr, Ys_tr, w, num_ex))
        est_err_te_1000.append(estimate_multinomial_logreg_error(Xs_te, Ys_te, w, num_ex))



    plt.plot(range(numberIter//monitorFreq+1), loss_np_te)
    plt.savefig("results/entire_ds_loss_test_"+str(numberIter)+".png")
    plt.close()

    plt.plot(range(numberIter//monitorFreq+1), error_np_te)
    plt.savefig("results/entire_ds_error_test_"+str(numberIter)+".png")
    plt.close()

    plt.plot(range(numberIter//monitorFreq+1), loss)
    plt.savefig("results/entire_ds_loss_train_"+str(numberIter)+".png")
    plt.close()

    plt.plot(range(numberIter//monitorFreq+1), error)
    plt.savefig("results/entire_ds_error_train_"+str(numberIter)+".png")
    plt.close()

    plt.plot(range(numberIter//monitorFreq+1), est_err_tr_100)
    plt.savefig("results/subsample_100_estimated_err_train_"+str(numberIter)+".png")
    plt.close()

    plt.plot(range(numberIter//monitorFreq+1), est_err_tr_1000)
    plt.savefig("results/subsample_1000_estimated_err_train_"+str(numberIter)+".png")
    plt.close()

    plt.plot(range(numberIter//monitorFreq+1), est_err_te_100)
    plt.savefig("results/subsample_100_estimated_err_test_"+str(numberIter)+".png")
    plt.close()

    plt.plot(range(numberIter//monitorFreq+1), est_err_te_1000)
    plt.savefig("results/subsample_1000_estimated_err_test_"+str(numberIter)+".png")
    plt.close()

    print("\nPart 4 complete.\n")
