import os
import numpy as np
from numpy import random
import scipy
import copy
import matplotlib
import mnist
import pickle
import time

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
        Xs_tr = np.ascontiguousarray(Xs_tr)
        Ys_tr = np.ascontiguousarray(Ys_tr)
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
    yHat = softmax(np.matmul(W, X_batch.T), axis=0) - Y_batch.T
    ans = np.matmul(yHat, X_batch) + gamma * W
    return ans / batchSize


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
    yHat = softmax(np.dot(W, Xs), axis=0).T
    count = 0
    for i in range(len(Ys)):
        pred = np.argmax(yHat[i])
        if Ys[i, pred] != 1:
            count += 1
    return count / len(Ys)


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
    T = num_epochs * Xs.shape[1]
    params = []
    for i in range(T):
        if i % monitor_period == 0:
            params.append(W)
        index = np.random.randint(0, Xs.shape[1])
        W = W - alpha * (multinomial_logreg_grad_i(Xs, Ys, [index], gamma, W))
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
            if j % monitor_period == 0:
                params.append(W)
            W = W - alpha * (multinomial_logreg_grad_i(Xs, Ys, [j], gamma, W))
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
    T = (num_epochs * Xs.shape[1]) // B
    for i in range(T):
        if i % monitor_period == 0:
            params.append(W)
        ii = []
        for j in range(B):
            ii.append(np.random.randint(0, Xs.shape[1]))
        W = W - alpha * (multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W))
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
def sgd_minibatch_sequential_scan(
    Xs, Ys, gamma, W, alpha, B, num_epochs, monitor_period
):
    # TODO students should implement this
    params = []
    for t in range(num_epochs):
        for j in range(Xs.shape[1] // B):
            if j % monitor_period == 0:
                params.append(W)
            ii = [(j * B + i) for i in range(B)]
            W = W - alpha * (multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W))
    params.append(W)
    return params


def run_sgd(pickle_file, algorithm_number, sgd_fn, sgd_args):
    print(f"\n{DIVIDER}\n")
    if pickle_file and os.path.isfile(pickle_file):
        print(
            f"Algorithm {algorithm_number} weights exist at {pickle_file}. SGD algo#{algorithm_number} skipped."
        )
        W = pickle.load(open(pickle_file, "rb"))
        print(f"Algorithm {algorithm_number} params loaded")
    else:
        print(f"Running Algorithm {algorithm_number} ...")
        W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
        # Call sgd func here
        # ----- UNTESTED
        sgd_args["W"] = W
        W = sgd_fn(**sgd_args)
        # ----- UNTESTED
        print(f"Algorithm {algorithm_number} complete.")
        if pickle_file:
            print(f"Dumping params to {pickle_file} ...")
            pickle.dump(W, open(pickle_file, "wb"))
            print(f"Dumping complete.")
    return W


def get_error(Xs, Ys, params):
    errors = []
    for w in params:
        errors.append(multinomial_logreg_error(Xs, Ys, w))
    return errors


def obtain_tr_te_errs(X_tr, Y_tr, X_te, Y_te, params, algo_num):
    print(f"\nCalculating training and testing errors for {algo_num}... ")
    ret = get_error(X_tr, Y_tr, params), get_error(X_te, Y_te, params)
    print(f"Error collection complete. ")
    return ret


def print_runtime(N, algo_num, sgd_fn, sgd_fn_args):
    start = time.time()
    for i in range(N):
        W1 = run_sgd("", algo_num, sgd_fn, sgd_fn_args)
    end = time.time()
    print(f"Avg Runtime for algorithm {algo_num} is : {(end-start)/N}")


def hyperParamTuning(num_epochs, alphas, algo_arg, sgd_type, type_number, batch_sizes=[]):
    name = f"w{str(type_number)}tr"
    print(f'\n Minimum error from original alpha: {min(error_dict[name])} for algorith {type_number}')
    algo_arg["num_epochs"] = num_epochs
    print(f'Performing hyperparam tuning with num_epochs={algo_arg["num_epochs"]} and alphas={alphas} for algorith {type_number}')
    minErrors = []
    for i in alphas:
        algo_arg["alpha"] = i
        if batch_sizes:
            for b in batch_sizes:
                algo_arg["B"] = b
                W = run_sgd("", type_number, sgd_type, algo_arg)
                minErrors.append(min(get_error(Xs_tr, Ys_tr, W)))
        else:
            W = run_sgd("", type_number, sgd_type, algo_arg)
            minErrors.append(min(get_error(Xs_tr, Ys_tr, W)))
    if batch_sizes:
        alph_batches = [[x,y] for x in alphas for y in batch_sizes]
        print(f'\n(alpha, min_error, batchSize using that alpha and batch size)= {[a + [minErrors[i]] for i, a in enumerate(alph_batches)]}\n')
    else:
        print(f'\n(alpha, min_error using that alpha)= {list(zip(alphas, minErrors))}\n')

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    print("Shape of initial X:", Xs_tr.shape)
    print("Shape of initial Y:", Ys_tr.shape)
    DIVIDER = "#" * 20
    pickle_file_dir = "pickle_files/"
    algo_1_pickle_file = pickle_file_dir + "algo1.pickle"
    algo_2_pickle_file = pickle_file_dir + "algo2.pickle"
    algo_3_pickle_file = pickle_file_dir + "algo3.pickle"
    algo_4_pickle_file = pickle_file_dir + "algo4.pickle"

    # Create pickle folder if not exists already
    if not os.path.isdir(pickle_file_dir):
        print("Pickle folder does not exist. Creating ...")
        os.makedirs(pickle_file_dir)
        print(f"Created {pickle_file_dir}.")

    # Hyperparams for sgd algo 1 and 2
    gamma = 0.0001
    alpha = 0.001
    num_epochs = 10
    monitor_period = 6000
    algo_1_2_args = {
        "Xs": Xs_tr,
        "Ys": Ys_tr,
        "gamma": gamma,
        "alpha": alpha,
        "num_epochs": num_epochs,
        "monitor_period": monitor_period,
    }
    W_1 = run_sgd(algo_1_pickle_file, 1, stochastic_gradient_descent, algo_1_2_args)
    assert(len(W_1) == ((num_epochs * Xs_tr.shape[1]) // monitor_period) + 1)


    W_2 = run_sgd(algo_2_pickle_file, 2, sgd_sequential_scan, algo_1_2_args)
    assert(len(W_2) == ((num_epochs * Xs_tr.shape[1]) // monitor_period) + 1)

    # Hyperparams for sgd algo 1 and 2
    gamma = 0.0001
    alpha = 0.05
    num_epochs = 10
    monitor_period = 100
    B = 60
    algo_3_4_args = {
        "Xs": Xs_tr,
        "Ys": Ys_tr,
        "gamma": gamma,
        "alpha": alpha,
        "num_epochs": num_epochs,
        "monitor_period": monitor_period,
        "B": B,
    }

    W_3 = run_sgd(algo_3_pickle_file, 3, sgd_minibatch, algo_3_4_args)
    assert(len(W_3) == ((num_epochs * Xs_tr.shape[1]) // (monitor_period * B)) + 1)

    W_4 = run_sgd(algo_4_pickle_file, 4, sgd_minibatch_sequential_scan, algo_3_4_args)
    assert(len(W_4) == ((num_epochs * Xs_tr.shape[1]) // (monitor_period * B)) + 1)

    part_1_errs_pickle = pickle_file_dir + "part1errs.pickle"
    if not os.path.isfile(part_1_errs_pickle):
        W_1_tr_err, W_1_te_err = obtain_tr_te_errs(Xs_tr, Ys_tr, Xs_te, Ys_te, W_1, 1)
        W_2_tr_err, W_2_te_err = obtain_tr_te_errs(Xs_tr, Ys_tr, Xs_te, Ys_te, W_2, 2)
        W_3_tr_err, W_3_te_err = obtain_tr_te_errs(Xs_tr, Ys_tr, Xs_te, Ys_te, W_3, 3)
        W_4_tr_err, W_4_te_err = obtain_tr_te_errs(Xs_tr, Ys_tr, Xs_te, Ys_te, W_4, 4)
        error_dict = {
            "w1tr": W_1_tr_err,
            "w1te": W_1_te_err,
            "w2tr": W_2_tr_err,
            "w2te": W_2_te_err,
            "w3tr": W_3_tr_err,
            "w3te": W_3_te_err,
            "w4tr": W_4_tr_err,
            "w4te": W_4_te_err,
        }

        print(f"\nDumping errors to {part_1_errs_pickle} ...")
        pickle.dump(error_dict, open(part_1_errs_pickle, "wb"))
        print(f"Dumping complete.")

    else:
        error_dict = pickle.load(open(part_1_errs_pickle, "rb"))


    figures_dir = "Figures/"
    if not os.path.isdir(figures_dir):
        print("Figures folder does not exist. Creating ...")
        os.makedirs(figures_dir)
        print(f"Created {figures_dir}.")
    plt.yticks(np.arange(0.02,0.2,0.05))
    plt.gca().set_ylim([0,0.25])
    plt.plot(range(101), error_dict["w1tr"])
    plt.plot(range(101), error_dict["w2tr"])
    plt.plot(range(101), error_dict["w3tr"])
    plt.plot(range(101), error_dict["w4tr"])
    plt.savefig(figures_dir + "sgd_tr_part_1" + ".png")
    plt.close()

    plt.plot(range(101), error_dict["w1te"])
    plt.plot(range(101), error_dict["w2te"])
    plt.plot(range(101), error_dict["w3te"])
    plt.plot(range(101), error_dict["w4te"])
    plt.savefig(figures_dir + "sgd_te_part_1" + ".png")
    plt.close()


    # ----- PART 2
    # For at least three different algorithm configurations you explored in this Part, plot the resulting error against the number of epochs in two figures, one for Training error and one for Test error, just as you did for the evaluation in Part 1.
    # If you found hyperparameters that improved the performance in Steps 2, 3, and 4, use those hyperparameters for these figures.


    # ----- SUBPART 2
    alphas = [10**-2, 2.5*10**-3, 5*10**-3, 7.5*10**-3]
    hyperParamTuning(10, alphas, algo_1_2_args, stochastic_gradient_descent, 1)

    # ----- SUBPART 3
    alphas = [0.01, 0.02, 0.03]
    hyperParamTuning(5, alphas, algo_1_2_args, stochastic_gradient_descent, 1)

    # ----- SUBPART 4
    # TODO: Batch size tuning ? ?
    alphas = [0.1*i for i in range(4,9)]
    batch_sizes = [100, 120, 150]
    hyperParamTuning(5, alphas, algo_3_4_args, sgd_minibatch_sequential_scan, 4, batch_sizes)

    # ----- SUBPART 5
    args_1 = args_2 = copy.copy(algo_1_2_args)

    alpha1 = 0.005
    args_1["alpha"] = alpha1
    args_1["num_epochs"] = 10
    W1_10 = run_sgd("", 1, stochastic_gradient_descent, args_1)
    E1_10_tr, E1_10_te = obtain_tr_te_errs(Xs_tr, Ys_tr, Xs_te, Ys_te, W1_10, 1)
    plt.plot(range(101), E1_10_tr)
    plt.plot(range(101), E1_10_te)
    plt.savefig(figures_dir + "sgd_1" + str(args_1["num_epochs"]) + "epochs.png")
    plt.close()

    alpha2 = 0.01
    args_2["alpha"] = alpha2
    args_2["num_epochs"] = 5
    W1_5 = run_sgd("", 1, stochastic_gradient_descent, args_2)
    E1_5_tr, E1_5_te = obtain_tr_te_errs(Xs_tr, Ys_tr, Xs_te, Ys_te, W1_5, 1)
    plt.plot(range(101), E1_5_tr)
    plt.plot(range(101), E1_5_te)
    plt.savefig(figures_dir + "sgd_1" + str(args_2["num_epochs"]) + "epochs.png")
    plt.close()

    args_3 = copy.copy(algo_3_4_args)
    alpha3 = 0.70
    args_3["alpha"] = alpha3
    args_3["num_epochs"] = 5
    W3 = run_sgd("", 4, sgd_minibatch_sequential_scan, args_3)
    E3_tr, E3_te = obtain_tr_te_errs(Xs_tr, Ys_tr, Xs_te, Ys_te, W_3, 4)
    plt.plot(range(101), E3_tr)
    plt.plot(range(101), E3_te)
    plt.savefig(figures_dir + "sgd_3" + str(args_3["num_epochs"]) + ".png")
    plt.close()

    # ----- PART 3
    N = 5
    print_runtime(N, 1, stochastic_gradient_descent, algo_1_2_args)
    print_runtime(N, 2, sgd_sequential_scan, algo_1_2_args)
    print_runtime(N, 3, sgd_minibatch, algo_3_4_args)
    print_runtime(N, 4, sgd_minibatch_sequential_scan, algo_3_4_args)
