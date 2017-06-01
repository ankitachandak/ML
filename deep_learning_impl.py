import numpy as np
from numpy.matlib import repmat
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
import time

def preprocess(xTr, xTe):
    """
    Preproces the data to make the training features have zero-mean and
    standard-deviation 1
    OUPUT:
        xTr - nxd training data
        xTe - mxd testing data
    OUPUT:
        xTr - pre-processed training data
        xTe - pre-processed testing data
        s,m - standard deviation and mean of xTr
            - any other data should be pre-processed by x-> (x-m)/s
    (The size of xTr and xTe should remain unchanged)
    """
    
    ntr, d = xTr.shape
    nte, _ = xTe.shape
    
    m = xTr.mean(axis=0)
    s = xTr.std(axis=0)
    xTr = (xTr-m)/s
    xTe = (xTe-m)/s

    return xTr, xTe, s, m

def get_transition_func(transtype):
    """
    Given the type, gets a specific transition function
    INPUT:
        transtype - "sigmoid", "tanh", "ReLU", "sin"
    OUTPUT:
        trans_func - transition function (function)
        trans_func_der - derivative of the transition function (function)

    (type must be one of the defined transition functions)
    """
    
    assert transtype in ["sigmoid", "tanh", "ReLU","sin"]
    if transtype == "sin":   
        trans_func = lambda z: np.sin(z)
        trans_func_der = lambda z: np.cos(z)
    elif transtype == "sigmoid":
        trans_func = lambda z: 1.0/(1.0 + np.exp(-z))
        trans_func_der = lambda z: np.exp(-z)/ np.square(1.0 + np.exp(-z))
    elif transtype == "tanh":
        trans_func = lambda z: (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
        trans_func_der = lambda z: 1 - ((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))**2
    elif transtype == "ReLU":
#         trans_func = lambda z: np.maximum(0, z)
        trans_func = lambda z: z * (z > 0)
        trans_func_der = lambda z: z > 0
#         trans_func_der = lambda z: 1 if z >= 0 else 0
    
    return trans_func, trans_func_der

def numericalgradient(fun,x,e):
    dh = 0
    nx = x    # copy the weight vector
    nx += e  # perturn dimension i
    l1 = fun(nx) # compute loss
    nx -= 2*e # perturm dimension i again
    l2 = fun(nx) # compute loss
    dh = (l1 - l2)/(2*e) # the gradient is the slope of the loss
    return dh

def initweights(specs):
    """
    Given a specification of the neural network, output a random weight array
    INPUT:
        specs - array of length m+1
    
    OUTPUT:
        weights - array of length m, each element is a matrix
            where size(weights[i]) = (specs[i], specs[i+1])
    """
    weights = []
    for i in range(len(specs) - 1):
        weights.append(np.random.randn(specs[i], specs[i+1]))
    return weights

def forward_pass(weights, xTr, trans_func):
    """
    INPUT:
        weights - weights (cell array of length m)
        xTr - nxd matrix (each row is an input vector)
        trans_func - transition function to apply for inner layers
    
    OUTPUTS:
        A, Z - result of forward pass (cell array of length m+1)
    
    Hint:
        Make sure A[0]=Z[0]=xTr and A[m] = Z[m] (Why?)
    """
    A = [xTr]
    Z = [xTr]
    for i in range(1,len(weights)+1):
        A.append(np.dot(Z[i-1] , weights[i-1]))
        Z.append(trans_func(A[i]))
    Z[len(weights)] = A[len(weights)]
    return A, Z

def compute_loss(Z, yTr):
    """
    INPUT:
        Z - output of forward pass (cell array of length m+1)
        yTr - array of length n
    
    OUTPUTS:
        loss - the average squared loss obtained with Z and yTr (scalar)
    """
    
    delta = Z[-1].flatten() - yTr.flatten()
    n = len(yTr)
    loss = 0

    loss = np.divide(np.float(np.sum(np.square(delta))), 2*n)

    return loss

def backprop(weights, A, Z, yTr, delta_f):
    """
    INPUT:
        weights - weights (cell array of length m)
        A - output of forward pass (cell array of length m+1)
        Z - output of forward pass (cell array of length m+1)
        yTr - array of length n
        delta_f - derivative of transition function to apply for inner layers
    
    OUTPUTS:
        gradient - the gradient at w (cell array of length m)
    """
    
    yTr = yTr.reshape(-1,1)
    n,_ = yTr.shape
    delta = (Z[-1].flatten() - yTr.flatten()).reshape(-1, 1)
    # compute gradient with back-prop
    delta = delta/n
    gradients = []
    for i in range(len(weights)-1, -1, -1):
        gradients.insert(0, np.dot(Z[i].T, delta))
        delta = np.dot(delta, weights[i].T) * delta_f(A[i])
        
    return gradients

def plot_results(x, y, Z, losses):
    fig, axarr = plt.subplots(1, 2)
    fig.set_figwidth(12)
    fig.set_figheight(4)

    axarr[0].plot(x, y)
    axarr[0].plot(x, Z[-1].flatten())
    axarr[0].set_ylabel('$f(x)$')
    axarr[0].set_xlabel('$x$')
    axarr[0].legend(['Actual', 'Predicted'])

    axarr[1].semilogy(losses)
    axarr[1].title.set_text('Loss')
    axarr[1].set_xlabel('Epoch')

    plt.show()

def adagrad_weights_adjust():
    # training data
    x = np.arange(0, 5, 0.1)
    y = (x ** 2 + 10*np.sin(x))
    x2d = np.concatenate([x, np.ones(x.shape)]).reshape(2, -1).T

    # transition function
    f, delta_f = get_transition_func("ReLU")

    # initialize weights, historical gradients, losses
    weights = initweights([2,200,1])
    losses = np.zeros(M)

    hist_grad = np.copy(weights)
    for j in range(len(weights)):
        hist_grad[j] = hist_grad[j] * 0

    alpha = 0.02
    M = 10000
    beta = 0.8
    eps = 1e-6

    losses = np.zeros(M)
    t0 = time.time()
    for i in range(M):
        f, delta_f = get_transition_func("ReLU")
        A, Z = forward_pass(weights, x2d, f)
        losses[i] = compute_loss(Z,y)
        gradients = backprop(weights,A,Z,y,delta_f)
        for j in range(len(weights)):
            hist_grad[j] += gradients[j] ** 2
            adj_grad = gradients[j] / (eps + np.sqrt(hist_grad[j]))
            weights[j] -= alpha * adj_grad
    t1 = time.time()
    plot_results(x, y, Z, losses)

def plot_samples():
	fig, axarr = plt.subplots(1, 4, sharex=True, sharey=True)
    fig.set_figwidth(15)
    fig.set_figheight(3)
    names = ["sigmoid","tanh","ReLU","sin"]

    for idx, name in enumerate(names):
        # plot stuff
        f, delta_f = get_transition_func(name)
        x = np.arange(-5, 5, 0.1)
        axarr[idx].plot(x, f(x))
        axarr[idx].axis([-5,5,-1,1])
        axarr[idx].title.set_text(name)
        axarr[idx].grid(True)
        
        # check gradients
        print("%s gradient check at x=1: " % name, end='')
        dh = numericalgradient(f,1,1e-5)
        dy = delta_f(1)
        num = np.linalg.norm(dh-dy)
        denom = np.linalg.norm(dh+dy)
        graderror = num/denom if denom != 0 else 0
        if graderror < 1e-10:
            print("passed ", end='')
        else:
            print("FAILED ", end='')
        print('at x=-1: ', end='')
        dh2 = numericalgradient(f,-1,1e-5)
        dy2 = delta_f(-1)
        num = np.linalg.norm(dh2-dy2)
        denom = np.linalg.norm(dh2+dy2)
        graderror += num/denom if denom != 0 else 0
        if graderror < 1e-10:
            print("passed")
        else:
            print("FAILED")
