import numpy as np
import sys
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import mode
import time
import os

def loaddata(filename):
    """
    Returns xTr,yTr,xTe,yTe
    xTr, xTe are in the form nxd
    yTr, yTe are in the form nx1
    """
    data = loadmat(filename)
    xTr = data["xTr"]; # load in Training data
    yTr = np.round(data["yTr"]); # load in Training labels
    xTe = data["xTe"]; # load in Testing data
    yTe = np.round(data["yTe"]); # load in Testing labels
    return xTr.T,yTr.T,xTe.T,yTe.T

def plotfaces(X, xdim=38, ydim=31, ):
    n, d = X.shape
    f, axarr = plt.subplots(1, n, sharey=True)
    f.set_figwidth(10 * n)
    f.set_figheight(n)
    
    if n > 1:
        for i in range(n):
            axarr[i].imshow(X[i, :].reshape(ydim, xdim).T, cmap=plt.cm.binary_r)
    else:
        axarr.imshow(X[0, :].reshape(ydim, xdim).T, cmap=plt.cm.binary_r)

def l2distance(X, Z=None):
    if Z is None:
        Z = X
    G = innerproduct(X,Z)
    x,z =  G.shape
    x1= np.dot(np.diagonal(innerproduct(X)).reshape(x,1), np.ones((1, z)))
    z1= np.dot(np.diag(innerproduct(Z)).reshape(z,1), np.ones((1,x)))
    return np.sqrt(x1 + z1.T - 2*innerproduct(X,Z))

def findknn(xTr,xTe,k) :  
    dist = l2distance(xTe, xTr)
    asd = np.argsort(dist)
    indices = asd[:,:k].T
    sd = np.sort(dist)
    dists = sd[:,:k].T

    return indices, dists
def analyze(kind,truth,preds):
    """
    function output=analyze(kind,truth,preds)         
    Analyses the accuracy of a prediction
    Input:
    kind='acc' classification error
    kind='abs' absolute loss
    (other values of 'kind' will follow later)
    """
    
    truth = truth.flatten()
    preds = preds.flatten()
    
    if kind == 'abs':
        loss   = np.absolute(np.subtract(truth, preds))
        output = np.divide(np.float(np.sum(loss)), np.size(truth))
    elif kind == 'acc':
        correct = np.equal(truth, preds)
        output  = np.divide(np.float(np.sum(correct)), np.size(truth))
    
    return output

def knnclassifier(xTr,yTr,xTe,k):
    """
    function preds=knnclassifier(xTr,yTr,xTe,k);
    
    k-nn classifier 
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    
    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    """
    if k > len(xTr):
        k = len(xTr)
    indices, distances = findknn(xTr,xTe,k)
    yTr = np.array(yTr)
    a = yTr[indices]
    preds = mode(a, axis=0)
    return preds[0].flatten()

def competition(xTr,yTr,xTe):
    ind = 2*len(xTr)/3
    xTraining = xTr[:ind,:]
    yTraining = yTr[:ind,:]
    xValid = xTr[ind:,:]
    yValid = yTr[ind:,:]
    acc = 0
    kFinal = 1
    for k in range(1,6):
        preds = knnclassifier(xTraining, yTraining, xValid, k)
        accTmp = analyze("acc", yValid, preds)
        print accTmp
        if acc < accTmp:
            print k
            acc = accTmp
            kFinal = k
    preds = knnclassifier(xTr, yTr, xTe, k)
    return preds

print("Face Recognition: (1-nn)")
xTr,yTr,xTe,yTe=loaddata("Documents/startercode/faces.mat") # load the data
t0 = time.time()
preds = knnclassifier(xTr,yTr,xTe,1)
result=analyze("acc",yTe,preds)
t1 = time.time()
print("You obtained %.2f%% classification acccuracy in %.4f seconds\n" % (result*100.0,t1-t0))

print("Handwritten digits Recognition: (3-nn)")
xTr,yTr,xTe,yTe=loaddata("Documents/startercode/digits.mat"); # load the data
t0 = time.time()
preds = knnclassifier(xTr,yTr,xTe,5)
result=analyze("acc",yTe,preds)
t1 = time.time()
print("You obtained %.2f%% classification acccuracy in %.4f seconds\n" % (result*100.0,t1-t0))

