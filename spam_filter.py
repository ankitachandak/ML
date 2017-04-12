import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from numpy.matlib import repmat
import sys
from scipy.io import loadmat
import time

%matplotlib inline

import re
import string
feature_dimension = 2048
def extractfeaturescomp(path, B):
    '''
    INPUT:
    path : file path of email
    B    : dimensionality of feature vector
    
    OUTPUTS:
    x    : B dimensional vector
    '''
    x = np.zeros(B)
    prepositions = ['a', 'all','always','the','are', 'is', 'and', 'were', 'or','on','also', 'an', 'am', 'for', 
                    'by', 'do', 'does', 'upon', 'b','c','d','e','f','g','h','j','k','l','m','n','o','p','q','r',
                    's','t','u','v','w','x','y','z', 'be', 'her','she', 'like']
    p = re.compile('\d+')
    with open(path, 'r') as femail:
        email = femail.read()
        # breaks for non-ascii characters
#         email = email.split('Message-ID')[-1]
        tokens = email.split()
        for i in range(len(tokens)):
            token = tokens[i]
#             token = token.strip(',.;!?:\'\" ')
            token = token.translate(None, string.punctuation)
            if token.lower() not in prepositions and not p.match(token):
                x[hash(token) % B] += 1
                x[hash(token.upper()) % B] += 1
                if i < len(tokens)-1:
                    x[hash(tokens[i]+ " " + tokens[i+1]) % B] += 1
                    x[hash(tokens[i].upper()+ " " + tokens[i+1].upper()) % B] += 1
    return x


def loadspamdata(extractfeatures, B=512, path="../resource/lib/public/data_train/"):
    '''
    INPUT:
    extractfeatures : function to extract features
    B               : dimensionality of feature space
    path            : the path of folder to be processed
    
    OUTPUT:
    X, Y
    '''
    if path[-1] != '/':
        path += '/'
    
    with open(path + 'index', 'r') as f:
        allemails = [x for x in f.read().split('\n') if ' ' in x]
    
    xs = np.zeros((len(allemails), B))
    ys = np.zeros(len(allemails))
    for i, line in enumerate(allemails):
        label, filename = line.split(' ')
        # make labels +1 for "spam" and -1 for "ham"
        ys[i] = (label == 'spam') * 2 - 1
        xs[i, :] = extractfeatures(path + filename, B)
    print('Loaded %d input emails.' % len(ys))
    return xs, ys

X,Y = loadspamdata(extractfeaturesnaive)
print X.shape

# Split data into training and validation
n, d = X.shape
cutoff = int(np.ceil(0.8 * n))
# indices of training samples
xTr = X[:cutoff,:]
yTr = Y[:cutoff]
# indices of testing samples
xTv = X[cutoff:,:]
yTv = Y[cutoff:]

def ridge(w,xTr,yTr,lmbda):
    """
    INPUT:
    w     : d   dimensional weight vector
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    lmbda : regression constant (scalar)
    
    OUTPUTS:
    loss     : the total loss obtained with w on xTr and yTr (scalar)
    gradient : d dimensional gradient at w
    """
    n, d = xTr.shape
    loss = np.sum(np.dot((np.dot(w,xTr.T)-yTr).T,(np.dot(w,xTr.T)-yTr)))/n + lmbda*(np.dot(w,w.T))
    gradient = 2*np.dot(xTr.T, (np.dot(w,xTr.T)-yTr).T)/n + 2*lmbda*(w)

    return loss, gradient

def numericalgradient(fun,w,e):
    # get dimensionality
    d = len(w)
    # initialize numerical derivative
    dh = np.zeros(d)
    # go through dimensions
    for i in range(d):
        # copy the weight vector
        nw = w.copy()
        # perturb dimension i
        nw[i] += e
        # compute loss
        l1, temp = fun(nw)
        # perturb dimension i again
        nw[i] -= 2*e
        # compute loss
        l2, temp = fun(nw)
        # the gradient is the slope of the loss
        dh[i] = (l1 - l2) / (2*e)
    return dh


def checkgrad(fun,w,e):
    # evaluate symbolic gradient from fun()
    loss,dy = fun(w)
    # estimate gradient numerically from fun()
    dh = numericalgradient(fun,w,e)
    
    # ii = dy.argsort()
    ii = np.array([i for i in range(len(dy))])
    
    plt.figure(figsize=(10,6))
    plt.scatter([i for i in range(len(dy))], dh[ii], c='b', marker='o', s=60)
    plt.scatter([i for i in range(len(dy))], dy[ii], c='r', marker='.', s=50)
    plt.xlabel('Dimension')
    plt.ylabel('Gradient value')
    plt.legend(["numeric","symbolic"])
    
    # return the norm of the difference scaled by the norm of the sum
    return np.linalg.norm(dh - dy) / np.linalg.norm(dh + dy)

# set lmbda (λ) arbitrarily
lmbda = 0.1
# dimensionality of the input
_, d = xTr.shape
# evaluate loss on random vector
w = np.random.rand(d)
# the lambda function notation is an inline way to define a function with only a single argument.
ratio = checkgrad(lambda weight: ridge(weight,xTr,yTr,lmbda),w,1e-05)
print("The norm ratio is %.10f." % ratio)

def adagrad(func,w,alpha,maxiter,delta=1e-02):
    """
    INPUT:
    func    : function to minimize
              (loss, gradient = func(w))
    w       : d dimensional initial weight vector 
    alpha   : initial gradient descent stepsize (scalar)
    maxiter : maximum amount of iterations (scalar)
    delta   : if norm(gradient)<delta, it quits (scalar)
    
    OUTPUTS:
     
    w      : d dimensional final weight vector
    losses : vector containing loss at each iteration
    """
    
    losses = np.zeros(maxiter)
    eps = 1e-06
    
    c = 0
    gSqr = np.zeros(np.shape(w)[0])
    while c < maxiter:
        loss, gradient = func(w)
        losses[c] = loss
        gSqr = (gSqr + gradient * gradient) + eps
        gSqrt = np.sqrt(gSqr)
        w1 = w - (alpha)*(gradient/gSqrt)
        if np.linalg.norm(gradient) < delta:
            return w1, losses
        w = w1
        c = c+1
    return w, losses

#Visualization
_, d = xTr.shape
print lmbda
w, losses = adagrad(lambda weight: ridge(weight, xTr, yTr, 0.1), np.random.rand(d), 1, 1000)

plt.figure(figsize=(10,6))
plt.semilogy(losses, c='r', linestyle='-')
plt.xlabel("gradient updates")
plt.ylabel("loss")
plt.title("Adagrad convergence")
print("Final loss: %f" % losses[-1])


def linclassify(w,xTr):
    pred = np.dot(xTr, w.T)
    pred[pred[:] > 0] = 1
    pred[pred[:] <= 0] = -1
    return pred

# evaluate training accuracy
preds = linclassify(w,xTr)
trainingacc = np.mean(preds==yTr)
# evaluate testing accuracy
preds = linclassify(w,xTv)
validationacc = np.mean(preds==yTv)
print("Training accuracy %2.2f%%\nValidation accuracy %2.2f%%\n" % (trainingacc*100,validationacc*100))


def logistic(w,xTr,yTr):
    """
    INPUT:
    w     : d   dimensional weight vector
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    
    OUTPUTS:
    loss     : the total loss obtained with w on xTr and yTr (scalar)
    gradient : d dimensional gradient at w
    """
    n, d = xTr.shape
    wtr  = np.log(1 + np.exp(-yTr*np.dot(w, xTr.T)))
    loss = np.sum(wtr)
    denom =  1 + np.exp(yTr*np.dot(w, xTr.T))
    num = np.multiply(-xTr, yTr[:, np.newaxis])
    gradient = np.sum((num/denom[:, np.newaxis]), axis=0)
    return loss, gradient

    raise NotImplementedError('Your code goes here!')


# Gradient sanity check
_, d = xTr.shape
w = np.random.rand(d)
ratio = checkgrad(lambda weight: logistic(weight,xTr,yTr),w,1e-05)
print("The norm ratio is %.10f." % ratio)

w, losses = adagrad(lambda weight: logistic(weight, xTr, yTr), np.random.rand(d), 1, 1000)

# evaluate training accuracy
preds = linclassify(w,xTr)
trainingacc = np.mean(preds==yTr)
# evaluate testing accuracy
preds = linclassify(w,xTv)
validationacc = np.mean(preds==yTv)
print("Training accuracy %2.2f%%\nValidation accuracy %2.2f%%\n" % (trainingacc*100,validationacc*100))

def hinge(w,xTr,yTr,lmbda):
    """
    INPUT:
    w     : d   dimensional weight vector
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    lmbda : regression constant (scalar)
    
    OUTPUTS:
    loss     : the total loss obtained with w on xTr and yTr (scalar)
    gradient : d dimensional gradient at w
    """
    n, d = xTr.shape
    err = 1 - yTr*np.dot(w,xTr.T)
    x1 = np.multiply(xTr, yTr[:, np.newaxis])
    num = np.sum(np.where(err[:, np.newaxis] > 0,-x1,0), axis=0)

    loss = lmbda*np.dot(w,w.T) + np.sum(np.where(err > 0,err, 0))

    gradient = 2*lmbda*(w) + num

    return loss,gradient

w, losses = adagrad(lambda weight: hinge(weight, xTr, yTr, lmbda), np.random.rand(d), 1, 1000)

# evaluate training accuracy
preds = linclassify(w,xTr)
trainingacc = np.mean(preds==yTr)
# evaluate testing accuracy
preds = linclassify(w,xTv)
validationacc = np.mean(preds==yTv)
print("Training accuracy %2.2f%%\nValidation accuracy %2.2f%%\n" % (trainingacc*100,validationacc*100))
lmbda = 0.1
_, d = xTr.shape
w = np.random.rand(d)
ratio = checkgrad(lambda weight: hinge(weight,xTr,yTr,lmbda),w,1e-05)
print("The norm ratio is %.10f." % ratio)

w, losses = adagrad(lambda weight: hinge(weight, xTr, yTr, lmbda), np.random.rand(d), 1, 1000)

# evaluate training accuracy
preds = linclassify(w,xTr)
trainingacc = np.mean(preds==yTr)
# evaluate testing accuracy
preds = linclassify(w,xTv)
validationacc = np.mean(preds==yTv)
print("Training accuracy %2.2f%%\nValidation accuracy %2.2f%%\n" % (trainingacc*100,validationacc*100))


def trainspamfiltercomp(xTr, yTr):
    '''
    INPUT:
    xTr : nxd dimensional matrix (each row is an input vector)
    yTr : d   dimensional vector (each entry is a label)
    
    OUTPUTS:
    w : d dimensional vector for linear classification
    '''
    w = np.zeros(np.shape(xTr)[1])
    columns = (xTr != 0).sum(0)
    xTr = xTr * xTr/columns
    w, losses = adagrad(lambda weight: logistic(weight, xTr, yTr), w, 8, 1000)
    return w
