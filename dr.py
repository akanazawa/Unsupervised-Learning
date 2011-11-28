from numpy import *
from util import *
from pylab import *

import pdb

def pca(X, K):
    '''
    X is an N*D matrix of data (N points in D dimensions)
    K is the desired maximum target dimensionality (K <= min{N,D})

    should return a tuple (P, Z, evals)
    
    where P is the projected data (N*K) where
    the first dimension is the higest variance,
    the second dimension is the second higest variance, etc.


    the low dimensional space (i.e., P = X * Z).

    and evals, a K dimensional array of eigenvalues (sorted)
    '''
    
    N,D = X.shape

    # make sure we don't look for too many eigs!
    if K > N:
        K = N
    if K > D:
        K = D

    # first, we need to center the data
    ### TODO: YOUR CODE HERE
    mu = X.mean(0)
    # python does element wise operation automatically
    covar = dot((X-mu).T, X-mu)/N # need extra re-normalization
    #    mu = X.mean(0).reshape(1,D)
    #    one = ones(N).reshape(N,1)
    #    x_centered = X - dot(one, mu)
    #    covar = dot(x_centered.transpose() , x_centered)
    
    # next, compute eigenvalues of the data variance
    #    hint 1: look at 'help(pylab.eig)'
    #    hint 2: you'll want to get rid of the imaginary portion of the eigenvalues; use: real(evals), real(evecs)
    #    hint 3: be sure to sort the eigen(vectors,values) by the eigenvalues: see 'argsort', and be sure to sort in the right direction!
    #             
    ### TODO: YOUR CODE HERE

    evals,evecs = eig(covar)
    idx = argsort(real(evals))[::-1] # get the sort idx and reverse the order

    # sort, remove img components, and get the first K components
    evals = real(evals[idx])[0:K] 
    Z = real(evecs[idx])[0:K] # the projection matrix
    
    # idx = argsort(evals[::-1]) #ran this works for here, but in general i think it's
    # idx = argsort(evals); idx= idx[::-1]
    # ########TODO##########
    # Z = []
    # for k in range(K):
    #     Z.append( evecs[idx[k]])
    # # in matlab notation this is : Z = evecs[idx][0:K]
    
    P = dot(X-mu, Z) # should be on centered data i think
    #    P = dot(X, Z) 
    #    Z = array(Z)
    return (P, Z, evals)

def kpca(X, K, kernel):
    '''
    X is an N*D matrix of data (N points in D dimensions)
    K is the desired maximum target dimensionality (K <= min{N,D})
    kernel is a FUNCTION that computes K(x,z) for some desired kernel and vectors x,z

    should return a tuple (P, alpha, evals), where P and evals are
    just like in PCA, and alpha is the vector of alphas (mixing
    parameters) for the kernel PCA decomposition.
    '''

    ### TODO: YOUR CODE HERE
    N,D = X.shape
    mu = X.mean(0)
    X_c = X-mu # centered
    # We want to compute K = X_cX_c' where X_c = X-mu, the centered kernel matrix
    # then X_cX_c = (X-mu)(X-mu)' = XX' - XM' - MX' - MM
    # where M =  1/N * ones((N,N))*X
    # let K0 = XX', the uncentered kernel matrix
    # then XM' = 1/N*X(ones(N,N)*X)' = 1/N*XX'*ones(N,N) = K_0/N
    # similarly, MX' = K_0/N, and MM' = 1/N^2(K_0)
    # so D = K_0 - 2K_0/N + K_0/N^2

    K0 = zeros((N, N))
    for i in range(N):
        for j in range(N):
            K0[i][j] = kernel(X[i, :], X[j, :])
    
    covar = (K0 - (1/N)*K0 - K0*(1/N) + (1/N)*K0*(1/N)) 

    # we're solving the equation Kv = N lam v
    evals,evecs = eig(covar)

    idx = argsort(real(evals))[::-1] # get the sort idx and reverse the order
    # sort, remove img components, and get the first K components
    evals = real(evals[idx])[0:K]
    Z = real(evecs[idx])[0:K]
    alpha =real(evecs[idx]) #??
    P = dot(X-mu, Z)
    
    return (P, alpha, evals)
