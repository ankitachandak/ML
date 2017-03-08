import numpy as np
def hashfeatures(baby, B, FIX):
    v = np.zeros(B)
    for b in [baby, baby.upper()]:
        for m in range(FIX):
            featurestring = "prefix%s%s" %(m,b[:m])
            v[hash(featurestring) % B] = 1
            featurestring = "suffix%s%s" %(m,b[-1:-(m+1):-1])
            v[hash(featurestring) % B] = 1
    return v

def name2features2(filename, B=1280, FIX=3, LoadFile=True):
    """
    Output:
    X : n feature vectors of dimension B, (nxB)
    """
    # read in baby names
    if LoadFile:
        with open(filename, 'r') as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split('\n')
    n = len(babynames)
    FIX = len(min(babynames, key=len))
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = hashfeatures(babynames[i], B, FIX)
    return X

def genTrainFeatures(dimension=128, fix=3):
    """
    function [x,y]=genTrainFeatures
    
    This function calls the python script "name2features.py" 
    to convert names into feature vectors and loads in the training data. 
    
    
    Output: 
    x: n feature vectors of dimensionality d [d,n]
    y: n labels (-1 = girl, +1 = boy)
    """
    
    # Load in the data
    Xgirls = name2features("girls.train", B=dimension, FIX=fix)
    Xboys = name2features("boys.train", B=dimension, FIX=fix)
    X = np.concatenate([Xgirls, Xboys])
    
    # Generate Labels
    Y = np.concatenate([-np.ones(len(Xgirls)), np.ones(len(Xboys))])
    
    # shuffle data into random order
    ii = np.random.permutation([i for i in range(len(Y))])
    
    return X[ii, :], Y[ii]

def naivebayesPY(x,y):
    """
    function [pos,neg] = naivebayesPY(x,y);

    Computation of P(Y)
    Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)

    Output:
    pos: probability p(y=1)
    neg: probability p(y=-1)
    """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    y = np.concatenate([y, [-1,1]])
    n = len(y)
    pos = float(np.count_nonzero(y == 1))/n
    neg = float(np.count_nonzero(y == -1))/n
    return pos,neg

def naivebayesPXY(x,y):
    """
    function [posprob,negprob] = naivebayesPXY(x,y);
    
    Computation of P(X|Y)
    Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)
    
    Output:
    posprob: probability vector of p(x|y=1) (dx1)
    negprob: probability vector of p(x|y=-1) (dx1)
    """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    n, d = x.shape
    x = np.concatenate([x, np.ones((2,d))])
    y = np.concatenate([y, [-1,1]])
    n, d = x.shape
    
    ## fill in code here
    new = np.column_stack((x, y))
    girls = new[new[:,-1] == -1][:,:-1]
    boys = new[new[:,-1] == 1][:,:-1]
    gtotal = np.sum(girls)
    btotal = np.sum(boys)
    girls = np.sum(girls, axis=0)
    boys = np.sum(boys, axis=0)
    girls = girls/float(gtotal)
    boys = boys/float(btotal)
    return boys, girls

def naivebayes(x,y,xtest):
    """
    function logratio = naivebayes(x,y);
    
    Computation of log P(Y|X=x1) using Bayes Rule
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)
    xtest: input vector of d dimensions (1xd)
    
    Output:
    logratio: log (P(Y = 1|X=x1)/P(Y=-1|X=x1))
    """
    
    ## fill in code here
    posprob,negprob = naivebayesPXY(x,y)
    pos,neg = naivebayesPY(x,y)
    posprob = np.sum(np.log(posprob))
    negprob = np.sum(np.log(negprob))
    return posprob - negprob + pos - neg

def naivebayesCL(x,y):
    """
    function [w,b]=naivebayesCL(x,y);
    Implementation of a Naive Bayes classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)

    Output:
    w : weight vector of d dimensions
    b : bias (scalar)
    """
    
    n, d = x.shape
    ## fill in code here
    posprob,negprob = naivebayesPXY(x,y)
    pos,neg = naivebayesPY(x,y)
    return np.log(posprob) - np.log(negprob), np.log(pos) - np.log(neg)

def classifyLinear(x,w,b=0):
    """
    function preds=classifyLinear(x,w,b);
    
    Make predictions with a linear classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    w : weight vector of d dimensions
    b : bias (optional)
    
    Output:
    preds: predictions
    """
    
    ## fill in code here
    pred = np.dot(x, w.T) + b
    pred[pred[:] > 0] = 1
    pred[pred[:] <= 0] = -1
    return pred

w,b = naivebayesCL(X,Y)
print('Training error: %.2f%%' % (100 *(classifyLinear(X, w, b) != Y).mean()))

#Helper to check/play around the names
DIMS = 128
print('Loading data ...')
X,Y = genTrainFeatures(DIMS)
print('Training classifier ...')
w,b=naivebayesCL(X,Y)
error = np.mean(classifyLinear(X,w,b) != Y)
print('Training error: %.2f%%' % (100 * error))

while True:
    print('Please enter your name>')
    yourname = input()
    if len(yourname) < 1:
        break
    xtest = name2features2(yourname,B=DIMS,LoadFile=False)
    pred = classifyLinear(xtest,w,b)[0]
    if pred > 0:
        print("%s, I am sure you are a nice boy.\n" % yourname)
    else:
        print("%s, I am sure you are a nice girl.\n" % yourname)


