from numpy import *
from pylab import *

import util, datasets, hmm, dr, kernel
# test PCA on Gaussian data with a known covar matrix
Si = util.sqrtm(array([[3,2],[2,4]]))
x = dot(randn(1000,2), Si)
plot(x[:,0], x[:,1], 'b.')
dot(x.T,x) / real(x.shape[0])

# run PAC on this data
(P, Z, evals) = dr.pca(x, 2)
# project the data
x0 = dot(dot(x, Z[0,:]).reshape(1000,1), Z[0,:].reshape(1,2))
x1 = dot(dot(x, Z[1,:]).reshape(1000,1), Z[1,:].reshape(1,2))
plot(x[:,0], x[:,1], 'b.', x0[:,0], x0[:,1], 'r.', x1[:,0], x1[:,1], 'g.')

# on digits data
(X,Y) = datasets.loadDigits()
(P,Z,evals) = dr.pca(X, 784)
evals

#FOR WU2
N, D = X.shape
# plot normalized eigenvalues
evals = evals/norm(evals)
plot(evals, 'r.')

# find where we have 90% of the var
summed = cumsum(evals)
total = summed[-1]
within90 = (summed <= total*.90).nonzero()[0]
within90[-1]+1 # this is how far we have to go to include 90% of the total var
within95 = (summed <= total*.95).nonzero()[0]
within95[-1]+1

# draw the top 50 eigvectors
util.drawDigits(Z[1:50, :], arange(50)

# ---------- KPCA ---------- #
Si = util.sqrtm(array([[3,2],[2,4]]))
x = dot(randn(1000,2), Si)
(P, alpha, evals) = dr.kpca(x, 2, kernel.linear)
evals
# do with data where vanilla PCA fails
(a,b) = datasets.makeKPCAdata()
plot(a[:,0], a[:,1], 'b.', b[:,0], b[:,1], 'r.')

x = vstack((a,b))
(P,Z,evals) = dr.pca(x, 2)

Pa = P[0:a.shape[0],:]
Pb = P[a.shape[0]:-1,:]
plot(Pa[:,0], randn(Pa.shape[0]), 'b.', Pb[:,0], randn(Pb.shape[0]), 'r.')

# now use KCPA
(P,alpha,evals) = dr.kpca(x, 2, kernel.rbf1)
evals
Pa = P[0:a.shape[0],:]
Pb = P[a.shape[0]:-1,:]
plot(Pa[:,0], Pa[:,1], 'b.', Pb[:,0], Pb[:,1], 'r.')

####################
# HMM
####################
(a,b,pi) = datasets.getHMMData()
hmm.viterbi(array([0,1,1,2]), a, b, pi)
#array([0, 0, 0, 1])

hmm.viterbi(array([0,2,1,2]), a, b, pi)
#array([0, 1, 1, 1])

al = hmm.forward(array([0,1,1,2]), a, b, pi)
be = hmm.backward(array([0,1,1,2]), a, b, pi)
hmm.sanityCheck(al,be)

##########
# parameter re-estimation
al = hmm.forward(array([0,1,1,2]), a, b, pi)
be = hmm.backward(array([0,1,1,2]), a, b, pi)
(a_new, b_new, pi_new) = hmm.reestimate(array([0,1,1,2]), al, be, a, b, pi)
