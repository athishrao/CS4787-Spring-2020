#!/usr/bin/env python3
import os
import copy

# BEGIN THREAD SETTINGS this sets the number of threads used by numpy in the program (should be set to 1 for Parts 1 and 3)
implicit_num_threads = 2
os.environ["OMP_NUM_THREADS"] = str(implicit_num_threads)
os.environ["MKL_NUM_THREADS"] = str(implicit_num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(implicit_num_threads)
# END THREAD SETTINGS
import pickle
import numpy as np
import numpy
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot as plt
import threading
import time
from scipy.special import softmax
from tqdm import tqdm

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


# SOME UTILITY FUNCTIONS that you may find to be useful, from my PA3 implementation
# feel free to use your own implementation instead if you prefer
def multinomial_logreg_error(Xs, Ys, W):
    predictions = numpy.argmax(numpy.dot(W, Xs), axis=0)
    error = numpy.mean(predictions != numpy.argmax(Ys, axis=0))
    return error

def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W, dtype=np.float64):
    WdotX = numpy.dot(W, Xs[:,ii])
    expWdotX = numpy.exp(WdotX - numpy.amax(WdotX, axis=0), dtype=dtype)
    softmaxWdotX = expWdotX / numpy.sum(expWdotX, axis=0, dtype=dtype)
    return numpy.dot(softmaxWdotX - Ys[:,ii], Xs[:,ii].transpose()) / len(ii) + gamma * W
# END UTILITY FUNCTIONS


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
        numpy.random.seed(4787)
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



# SGD + Momentum (adapt from Programming Assignment 3)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    # TODO students should use their implementation from programming assignment 3
    # or adapt this version, which is from my own solution to programming assignment 3
    models = []
    (d, n) = Xs.shape
    V = numpy.zeros(W0.shape)
    W = W0
    # print("Running minibatch sequential-scan SGD with momentum")
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            ii = range(ibatch*B, (ibatch+1)*B)
            V = beta * V - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
            W = W + V
            # if ((ibatch+1) % monitor_period == 0):
            #     models.append(W)
    return models


# SGD + Momentum (No Allocation) => all operations in the inner loop should be a
#   call to a numpy.____ function with the "out=" argument explicitly specified
#   so that no extra allocations occur
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
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_noalloc(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO students should initialize the parameter vector W and pre-allocate any needed arrays here
    Y_temp = np.zeros((c,B))
    W_temp = np.zeros(W0.shape)
    amax_temp = np.zeros(B)
    softmax_temp = np.zeros((c,B))
    V = np.zeros(W0.shape)
    g = np.zeros(W0.shape)
    X_batch = []
    Y_batch = []
    for i in range(n // B):
        ii = [(i*B + j) for j in range(B)]
        X_batch.append(np.ascontiguousarray(Xs[:,ii]))
        Y_batch.append(np.ascontiguousarray(Ys[:,ii]))
    # print("Running minibatch sequential-scan SGD with momentum (no allocation)")
    for it in tqdm(range(num_epochs)):
        for i in range(int(n/B)):
            # ii = range(ibatch*B, (ibatch+1)*B)
            # TODO this section of code should only use numpy operations with the "out=" argument specified (students should implement this)
            np.matmul(W0, X_batch[i], out=Y_temp)

            # WdotX = numpy.dot(W0, Xs[:,ii])
            # expWdotX = numpy.exp(WdotX - numpy.amax(WdotX, axis=0), dtype=dtype)
            # softmaxWdotX = expWdotX / numpy.sum(expWdotX, axis=0, dtype=dtype)
            np.amax(Y_temp, axis=0, out=amax_temp)
            np.subtract(Y_temp, amax_temp, out=softmax_temp)
            np.exp(softmax_temp, out=softmax_temp)
            np.sum(softmax_temp, axis=0, out=amax_temp)
            np.divide(softmax_temp, amax_temp, out=softmax_temp)

            # numpy.dot(softmaxWdotX - Ys[:,ii], Xs[:,ii].transpose()) / len(ii) + gamma * W
            np.subtract(softmax_temp, Y_batch[i], out=Y_temp)
            # Y_temp = softmax(Y_temp, axis=0) - Y_batch[i]

            np.matmul(Y_temp, X_batch[i].T, out=W_temp)
            np.divide(W_temp, B, out=W_temp)
            np.multiply(gamma, W0, out=g)
            np.add(W_temp, g, out=g)
            # g = W_temp / B +  gamma * W0
            np.multiply(V, beta, out=V)
            np.multiply(g, alpha, out=g)
            # V = (beta * V) - (alpha * g)
            np.subtract(V, g, out=V)
            np.add(V, W0, out=W0)
            # W0 = W0 + V
    return W0


# SGD + Momentum (threaded)
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
# num_threads     how many threads to use
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_threaded(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, num_threads):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO perform any global setup/initialization/allocation (students should implement this)
    g = [W0 for i in range(num_threads)]
    Bt = B//num_threads

    W_temp1 = np.zeros(W0.shape)
    # construct the barrier object
    iter_barrier = threading.Barrier(num_threads + 1)

    # a function for each thread to run
    def thread_main(ithread):
        # TODO perform any per-thread allocations
        for it in range(num_epochs):
            W_temp = np.zeros(W0.shape)
            amax_temp = np.zeros(Bt)
            softmax_temp = np.zeros((c,Bt))
            for ibatch in range(int(n/B)):
                # TODO work done by thread in each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
                ii = range(ibatch*B + ithread*Bt, ibatch*B + (ithread+1)*Bt)
                iter_barrier.wait()
                np.dot(g[ithread], Xs[:,ii], out=softmax_temp)
                np.amax(softmax_temp, axis=0, out=amax_temp)
                np.subtract(softmax_temp, amax_temp, out=softmax_temp)
                np.exp(softmax_temp, out=softmax_temp)
                np.sum(softmax_temp, axis=0, out=amax_temp)
                np.divide(softmax_temp, amax_temp, out=softmax_temp)
                np.subtract(softmax_temp,  Ys[:,ii], out=softmax_temp)
                np.matmul(softmax_temp,  Xs[:,ii].T, out=W_temp)
                np.divide(W_temp, B, out=W_temp)
                np.multiply(gamma,  g[ithread], out=g[ithread])
                np.add(W_temp, g[ithread], out=g[ithread])
                # g[ithread] = multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W0)
                iter_barrier.wait()

    worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]

    for t in worker_threads:
        print("running thread ", t)
        t.start()

    print("Running minibatch sequential-scan SGD with momentum (%d threads)" % num_threads)
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            iter_barrier.wait()
            # TODO work done on a single thread at each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
            # W0 = W0 - alpha * (1/B) * np.sum(g)
            np.sum(g, axis=0, out=W_temp1)
            np.multiply(W_temp1, alpha/B, out=W_temp1)
            np.subtract(W0, W_temp1, out=W0)

            iter_barrier.wait()

    for t in worker_threads:
        t.join()

    # return the learned model
    return W0


# SGD + Momentum (No Allocation) in 32-bits => all operations in the inner loop should be a
#   call to a numpy.____ function with the "out=" argument explicitly specified
#   so that no extra allocations occur
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
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_noalloc_float32(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    Xs = Xs.astype(np.float32)
    Ys = Ys.astype(np.float32)
    W0 = W0.astype(np.float32)

    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO students should initialize the parameter vector W and pre-allocate any needed arrays here
    Y_temp = np.zeros((c,B), dtype=np.float32)
    W_temp = np.zeros(W0.shape, dtype=np.float32)
    amax_temp = np.zeros((B,), dtype=np.float32)
    softmax_temp = np.zeros((c,B), dtype=np.float32)
    V = np.zeros(W0.shape, dtype=np.float32)
    g = np.zeros(W0.shape, dtype=np.float32)
    X_batch = []
    Y_batch = []
    for i in range(n // B):
        ii = [(i*B + j) for j in range(B)]
        X_batch.append(np.ascontiguousarray(Xs[:,ii], dtype=np.float32))
        Y_batch.append(np.ascontiguousarray(Ys[:,ii], dtype=np.float32))
    print("Running minibatch sequential-scan SGD with momentum (no allocation)")
    for it in tqdm(range(num_epochs)):
        for i in range(int(n/B)):
            # ii = range(ibatch*B, (ibatch+1)*B)
            # TODO this section of code should only use numpy operations with the "out=" argument specified (students should implement this)
            np.matmul(W0, X_batch[i], out=Y_temp).astype(numpy.float32)
            np.amax(Y_temp, axis=0, out=amax_temp).astype(numpy.float32)
            np.subtract(Y_temp, amax_temp, out=softmax_temp, dtype=np.float32)
            np.exp(softmax_temp, out=softmax_temp, dtype=np.float32)
            np.sum(softmax_temp, axis=0, out=amax_temp, dtype=np.float32)
            np.divide(softmax_temp, amax_temp, out=softmax_temp, dtype=np.float32)
            np.subtract(softmax_temp, Y_batch[i], out=Y_temp, dtype=np.float32)
            np.matmul(Y_temp, X_batch[i].T, out=W_temp, dtype=np.float32)
            np.divide(W_temp, B, out=W_temp, dtype=np.float32)
            np.multiply(gamma, W0, out=g, dtype=np.float32)
            np.add(W_temp, g, out=g, dtype=np.float32)
            np.multiply(V, beta, out=V, dtype=np.float32)
            np.multiply(g, alpha, out=g, dtype=np.float32)
            np.subtract(V, g, out=V, dtype=np.float32)
            np.add(V, W0, out=W0, dtype=np.float32)
    return W0


# SGD + Momentum (threaded, float32)
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
# num_threads     how many threads to use
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_threaded_float32(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, num_threads):
    Xs = Xs.astype(np.float32)
    Ys = Ys.astype(np.float32)
    W0 = W0.astype(np.float32)
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # TODO perform any global setup/initialization/allocation (students should implement this)
    g = [W0 for i in range(num_threads)]
    Bt = B//num_threads

    W_temp1 = np.zeros(W0.shape).astype(np.float32)
    # amax_temp = np.zeros(Bt)
    # softmax_temp = np.zeros((c,Bt))
    # construct the barrier object
    iter_barrier = threading.Barrier(num_threads + 1)

    # a function for each thread to run
    def thread_main(ithread):
        W_temp = np.zeros(W0.shape, dtype=np.float32)
        amax_temp = np.zeros((Bt,), dtype=np.float32)
        softmax_temp = np.zeros((c,Bt), dtype=np.float32)
        # TODO perform any per-thread allocations
        for it in range(num_epochs):
            for ibatch in range(int(n/B)):
                # TODO work done by thread in each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
                ii = range(ibatch*B + ithread*Bt, ibatch*B + (ithread+1)*Bt)
                iter_barrier.wait()

                np.dot(g[ithread], Xs[:,ii], out=softmax_temp).astype(np.float32)
                np.amax(softmax_temp, axis=0, out=amax_temp).astype(np.float32)
                np.subtract(softmax_temp, amax_temp, out=softmax_temp, dtype=np.float32)
                np.exp(softmax_temp, out=softmax_temp, dtype=np.float32)
                np.sum(softmax_temp, axis=0, out=amax_temp, dtype=np.float32)
                np.divide(softmax_temp, amax_temp, out=softmax_temp, dtype=np.float32)
                np.subtract(softmax_temp,  Ys[:,ii], out=softmax_temp, dtype=np.float32)
                np.matmul(softmax_temp,  Xs[:,ii].T, out=W_temp, dtype=np.float32)
                np.divide(W_temp, B, out=W_temp, dtype=np.float32)
                np.multiply(gamma,  g[ithread], out=g[ithread], dtype=np.float32)
                np.add(W_temp, g[ithread], out=g[ithread], dtype=np.float32)

                iter_barrier.wait()

    worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]

    for t in worker_threads:
        print("running thread ", t)
        t.start()

    print("Running minibatch sequential-scan SGD with momentum (%d threads)" % num_threads)
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            iter_barrier.wait()
            # TODO work done on a single thread at each iteration; this section of code should primarily use numpy operations with the "out=" argument specified (students should implement this)
            # W0 = W0 - alpha * (1/B) * np.sum(g)
            np.sum(g, axis=0, out=W_temp1, dtype=np.float32)
            np.multiply(W_temp1, alpha/B, out=W_temp1, dtype=np.float32)
            np.subtract(W0, W_temp1, out=W0, dtype=np.float32)

            iter_barrier.wait()

    for t in worker_threads:
        t.join()

    # return the learned model
    return W0

def run_algo(algorithm_identifier, algo_args):
    algorithms = {
        "sgd_momen": (sgd_mss_with_momentum, "SGD with Momentum"),
        "sgd_momen_no_alloc": (sgd_mss_with_momentum_noalloc, "SGD with Momentum (no alloc)"),
        "sgd_momen_threaded": (sgd_mss_with_momentum_threaded, "SGD with Momentum (Threaded)"),
        "sgd_momen_no_alloc_fl32": (sgd_mss_with_momentum_noalloc_float32, "SGD with Momentum (no alloc, float 32)"),
        "sgd_momen_threaded_fl32": (sgd_mss_with_momentum_threaded_float32, "SGD with Momentum (Threaded, float 32)"),
    }
    algo_fn = algorithms[algorithm_identifier][0]
    print(f"\nRunning Algorithm {algorithm_identifier} ...")
    print_config(algo_args)
    d, n = algo_args["Xs"].shape
    c, n = algo_args["Ys"].shape
    W0 = np.zeros((c, d))
    algo_args["W0"] = W0
    start = time.time()
    model = algo_fn(**algo_args)
    end = time.time()
    time_taken = end - start
    print(f"Algorithm {algorithm_identifier} complete. Time taken: {time_taken}")
    return  time_taken, W0

def generatePlot(listOfElements, nameOfElements, batchSizes, batchNames):
    figures_dir = "Figures/"
    if not os.path.isdir(figures_dir):
        print("Figures folder does not exist. Creating ...")
        os.makedirs(figures_dir)
        print(f"Created {figures_dir}.")

    for n, element in enumerate(listOfElements):
        print((element))
        print()
        print((batchSizes))
        print()
        print(nameOfElements[n])
        print()
        print()
        plt.loglog(batchNames, (element), label=nameOfElements[n])
        # plt.xticks(batchSizes, batchNames)
    plt.xlabel("Batch Sizes")
    plt.ylabel("Time Taken")
    plt.legend(loc="upper right")

    plt.savefig(figures_dir + "Result" + ".png")
    return plt

def print_config(config):
    print("Current Configuration:")
    print("~" * 15)
    for k in config:
        if k != "Xs" and k != "Ys" and k!= "W0":
            print(f"{k}: {config[k]}")
    print("~" * 15)

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO: NEXT LINE IS ONLY FOR DEBUGGING, REMOVE ON SUBMISSION
    Xs_tr, Ys_tr = Xs_tr[:50], Ys_tr[:50]
    sgd_args = {
        "Xs": Xs_tr,
        "Ys": Ys_tr,
        "alpha": 0.1,
        "num_epochs": 20,
        "beta": 0.9,
        "gamma": 10 ** -4,
    }
    DIVIDER = "#" * 20

    # Comments from the documenttion
    # TODO FOR sgd_mss_with_momentum_noalloc
    # To validate your implementation, you should check that the output of this new function is close to the output of your original sgd_mss_with_momentum function.
    def checkSimilarity(original, altered, atol=10**-6):
        isSimilar = np.allclose(original, altered, rtol=1, atol=atol)
        print(f"Similarity of the two matrices : {isSimilar}")
        return isSimilar

    # TODO For threaded
    # In order to make sure that your cores are not overloaded, you should set the number of cores used implicitly by numpy back to 1 (allowing the cores to be used explicitly by your implementation).
    listOfElements = []
    batch_sizes = [8, 16, 30, 60, 200, 600, 3000]

    if not os.path.exists("part134(1).pickle") and implicit_num_threads==1:
        # ----- PART-1
        sgd_momen, sgd_momen_noalloc = [], []
        for batch_size in batch_sizes:
            sgd_args["B"] = batch_size
            time_alloc, W_alloc = run_algo("sgd_momen", sgd_args)
            sgd_momen += [time_alloc]
            time_no_alloc, W_no_alloc = run_algo("sgd_momen_no_alloc", sgd_args)
            sgd_momen_noalloc += [time_no_alloc]
            checkSimilarity(W_alloc, W_no_alloc)

        # ----- PART-3
        # TODO: Reset implicit numpy multithreading
        # e_threaded = explicitly_threaded
        e_threaded = []
        threaded_args = copy.copy(sgd_args)
        threaded_args["num_threads"] = 2
        for batch_size in batch_sizes:
            threaded_args["B"] = batch_size
            time_threaded, Ws = run_algo("sgd_momen_threaded", threaded_args)
            e_threaded += [time_threaded]

        # ----- PART-4 (1)
        # fl32_noalloc_e is for explicit threading
        fl32_noalloc_e, fl32_threaded = [], []
        for batch_size in batch_sizes:
            sgd_args["B"] = batch_size
            time_noalloc, W_no_alloc = run_algo("sgd_momen_no_alloc_fl32", sgd_args)
            fl32_noalloc_e += [time_noalloc]
            threaded_args["B"] = batch_size
            time_threaded, W_threaded = run_algo("sgd_momen_threaded_fl32", threaded_args)
            fl32_threaded += [time_threaded]
            checkSimilarity(W_threaded, W_no_alloc)

        listA = [sgd_momen, sgd_momen_noalloc, e_threaded, fl32_noalloc_e, fl32_threaded]
        pickle.dump(listA, open("part134(1).pickle", "wb"))

    elif os.path.exists("part134(1).pickle"):
        f1 =  open("part134(1).pickle", "rb")
        listOfElements134_1 = pickle.load(f1)
        # listofList = [sgd_momen, sgd_momen_noalloc, e_threaded, fl32_noalloc_e, fl32_threaded]
        sgd_momen = listOfElements134_1[0]
        sgd_momen_noalloc = listOfElements134_1[1]
        e_threaded = listOfElements134_1[2]
        fl32_noalloc_e = listOfElements134_1[3]
        fl32_threaded = listOfElements134_1[4]

    if not os.path.exists("part24(2).pickle") and implicit_num_threads==2:
        # ----- PART-2
        # TODO: Change the environ variables here
        # i_threaded = implicitly_threaded
        i_threaded, noalloc_i_threaded = [], []
        for batch_size in batch_sizes:
            sgd_args["B"] = batch_size
            time_alloc, W_alloc = run_algo("sgd_momen", sgd_args)
            i_threaded += [time_alloc]
            time_no_alloc, W_no_alloc = run_algo("sgd_momen_no_alloc", sgd_args)
            noalloc_i_threaded += [time_no_alloc]
            checkSimilarity(W_alloc, W_no_alloc)

        # ----- PART-4 (2)
        # TODO: Change the environ variables here to use implicit therading
        # fl32_noalloc_i is for implicit threading
        fl32_noalloc_i = []
        for batch_size in batch_sizes:
            sgd_args["B"] = batch_size
            time_fl32_noalloc_i, W = run_algo("sgd_momen_no_alloc_fl32", sgd_args)
            fl32_noalloc_i += [time_fl32_noalloc_i]

        listA = [i_threaded, noalloc_i_threaded, fl32_noalloc_i]
        pickle.dump(listA, open("part24(2).pickle", "wb"))

    elif os.path.exists("part24(2).pickle"):
        f2 =  open("part24(2).pickle", "rb")
        listOfElements24_2 = pickle.load(f2)

        # listofList = [i_threaded, noalloc_i_threaded, fl32_noalloc_i]
        i_threaded = listOfElements24_2[0]
        noalloc_i_threaded = listOfElements24_2[1]
        fl32_noalloc_i = listOfElements24_2[2]


    listOfElements = [sgd_momen, sgd_momen_noalloc, fl32_noalloc_e, i_threaded, noalloc_i_threaded, fl32_noalloc_i, e_threaded, fl32_threaded]

    nameOfElements = ["Baseline-64 Exp", "NMA-64 Exp", "NMA-32 Exp", "Baseline-64 Imp", "NMA-64 Imp", "NMA-32 Imp", "Multithreaded-64 Exp", "Multithreaded-32 Exp"]
    generatePlot(listOfElements, nameOfElements, range(len(batch_sizes)), batch_sizes)
