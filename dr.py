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

    Z is the projection matrix (D*K) that projects the data into
    the low dimensional space (i.e., P = X * Z).

    and evals, a K dimensional array of eigenvalues (sorted)
    '''
    
    N,D = X.shape

    # make sure we don't look for too many eigs!
    if K > N:
        K = N
    if K > D:
        K = D

    X_c = X - X.mean(0) # center the data
    covar = dot((X_c).T, X_c)/N # need extra re-normalization
    
    # next, compute eigenvalues of the data variance

	# HAL's Hints
    # +  hint 1: look at 'help(pylab.eig)'
    # +  hint 2: you'll want to get rid of the imaginary portion of the eigenvalues; use: real(evals), real(evecs)
    # +  hint 3: be sure to sort the eigen(vectors,values) by the eigenvalues: see 'argsort', and be sure to sort in the right direction!

    evals,evecs = eig(covar)

    idx = argsort(real(evals))[::-1] # get the sort idx and reverse the order
    idx = idx[0:K]

    # sort, remove imaginary components, and get the first K components
    evals = real(evals[idx])
    Z = real(evecs[:, idx]) # the projection matrix
    
	# Project the centered data    
    P = dot(X_c, Z)
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

    N,D = X.shape
    X_c = X - X.mean(0) # center the data

	# Compute K0
    K0 = zeros((N, N))
    for i in range(N):
        for j in range(N):
            K0[i][j] = kernel(X[i, :], X[j, :])
    
    # Kernel = (K0 - H*K0 - K0*H + H*K0*H)
	H = ones((N, N))/N
    Ker = (K0 - dot(H,K0) - dot(K0,H) + dot(dot(H,K0),H))

    # we're solving the equation Kv = N lam v
    evals,evecs = eig(Ker)
    # get the sort idx and reverse the order
    idx = argsort(real(evals))[::-1]
    idx = idx[0:K]

    # sort, remove img components, and get the first K components
    # need to normalize evals N*lamda*K*alpha = K*alpha
    evals = real(evals[idx]) / N
    alpha = real(evecs[:, idx])
    Z = zeros((N, K))

#	pdb.set_trace()

    # normalize alpha
    for i in range(K):
        Z[:,i:(i+1)] = alpha[:,i:(i+1)] / (sqrt(evals[i]))
        #Z[:,i:(i+1)] = alpha[:,i:(i+1)] / (sqrt(dot(alpha[:,i:(i+1)].T, alpha[:,i:(i+1)]) * evals[i])) # This works too but the previous line should be more efficient
        #print evals[i] * dot(Z[:,i:(i+1)].T, Z[:,i:(i+1)]) # Should be 1
        #print (dot(Z[:,i:(i+1)].T, dot(Ker, Z[:,i:(i+1)])) / N) # Should be 1

    P = dot(Ker, Z)

    return (P, Z, evals)
