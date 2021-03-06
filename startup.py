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
# x0 = dot(dot(x, Z[0,:]).reshape(1000,1), Z[0,:].reshape(1,2))
# x1 = dot(dot(x, Z[1,:]).reshape(1000,1), Z[1,:].reshape(1,2))

x0 = dot(dot(x, Z[:, 0]).reshape(1000,1), Z[:, 0].reshape(1,2))
x1 = dot(dot(x, Z[:, 1]).reshape(1000,1), Z[:, 1].reshape(1,2))

plot(x[:,0], x[:,1], 'b.', x0[:,0], x0[:,1], 'r.', x1[:,0], x1[:,1], 'g.')
axis((-8.0, 8.0, -8.0, 8.0))

# on digits data
(X,Y) = datasets.loadDigits()
(P,Z,evals) = dr.pca(X, 784)
#evals

#FOR WU2
N, D = X.shape
# plot normalized eigenvalues
evals = evals/sum(evals)

# find where we have 90% of the var
summed = cumsum(evals)
total = summed[-1] # should be ~1
within90 = (summed <= total*.90).nonzero()[0]
within90[-1]+1 # this is how far we have to go to include 90% of the total var
within95 = (summed <= total*.95).nonzero()[0]
within95[-1]+1

#FOR WU2 Figure a
plot(evals, 'r,')
axis((0.0, 800.0, -0.01, 0.1))

#FOR WU2 Figure b
plot(summed, 'r,')
axis((0.0, 800.0, 0.0, 1.1))
axhline(y=.9, xmin=0, xmax=(10.0/80.0), color='b')
axhline(y=.95, xmin=0, xmax=(15.5/80.0), color='b')
annotate('(81, 90.0)', xy=(81, .9), xytext=(151, .7), arrowprops=dict(facecolor='black', shrink=0.05, width=.5, headwidth=4),)
annotate('(135, 95.0)', xy=(135, .95), xytext=(205, .75), arrowprops=dict(facecolor='black', shrink=0.05, width=.5, headwidth=4),)

#FOR WU3 Figure draw the top 50 eigvectors
util.drawDigits(Z[:,0:49].T,arange(50))
#FOR WU3b
util.drawDigits(dot(P,Z.T),Y)
#FOR WU3c
(P,Z,evals) = dr.pca(X, 5)
util.drawDigits(dot(P,Z.T),Y)

# ---------- KPCA ---------- #
Si = util.sqrtm(array([[3,2],[2,4]]))
x = dot(randn(1000,2), Si)
(P, alpha, evals) = dr.kpca(x, 2, kernel.linear)
evals
# do with data where vanilla PCA fails
(a,b) = datasets.makeKPCAdata()
plot(a[:,0], a[:,1], 'b.', b[:,0], b[:,1], 'r.')

x = vstack((a,b))
x_c = (x-mean(x))/std(x)
(P,Z,evals) = dr.pca(x, 2)

Pa = P[0:a.shape[0],:]
Pb = P[a.shape[0]:,:]
plot(Pa[:,0], randn(Pa.shape[0]), 'b.', Pb[:,0], randn(Pb.shape[0]), 'r.')

# projection of the data with each eig vectors...
# x0 = dot(dot(x, Z[:, 0]).reshape(473,1), Z[:, 0].reshape(1,2))
# x1 = dot(dot(x, Z[:, 1]).reshape(473,1), Z[:, 1].reshape(1,2))
# pro0a = x0[0:a.shape[0],:]
# pro0b = x0[a.shape[0]:,:]
# plot( pro0a[:,0], 'b.', pro0b[:,0], 'r.', )

# now use KCPA
(P,alpha,evals) = dr.kpca(x, 2, kernel.rbf1)
#evals

Pa = P[0:a.shape[0],:]
Pb = P[a.shape[0]:,:]
plot(Pa[:,0], Pa[:,1], 'b.', Pb[:,0], Pb[:,1], 'r.')

plot(alpha[:, 0],'r.')
####################
# HMM
####################
(a,b,pi) = datasets.getHMMData()
hmm.viterbi(array([0,1,1,2]), a, b, pi)
#array([0, 0, 0, 1])

hmm.viterbi(array([0,2,1,2]), a, b, pi)
#array([0, 1, 1, 1])

###WU 8
# example 1
hmm.viterbi(array([0,1,1,1]), a, b, pi) # 0 0 0 0
hmm.viterbi(array([0,1,2,1]), a, b, pi) # 0 0 1 1


al = hmm.forward(array([0,1,1,2]), a, b, pi)
be = hmm.backward(array([0,1,1,2]), a, b, pi)
hmm.sanityCheck(al,be)

##########
# parameter re-estimation
al = hmm.forward(array([0,1,1,2]), a, b, pi)
be = hmm.backward(array([0,1,1,2]), a, b, pi)
(a_new, b_new, pi_new) = hmm.reestimate(array([0,1,1,2]), al, be, a, b, pi)

# >>> a_new
# array([[ 0.53662942,  0.46337058],
#        [ 0.39886289,  0.60113711]])
# >>> b_new
# array([[ 0.35001693,  0.55333559,  0.09664748],
#        [ 0.14235731,  0.44259786,  0.41504483]])
# >>> pi_new
# array([ 0.72574077,  0.27425923])

(a_em, b_em, pi_em, logProbs)=hmm.runEM(array([0,1,1,2]),2,3)

# >>> a_em
# array([[  2.54354295e-293,   1.00000000e+000],
#        [  1.00000000e+000,   5.67486661e-014]])
# >>> b_em
# array([[  5.00000000e-001,   5.00000000e-001,   1.49055507e-282],
#        [  0.00000000e+000,   5.00000000e-001,   5.00000000e-001]])
# >>> pi_em
# array([ 1.,  0.])

# the result says that the states always alternate
# the first state is always state 0
# ->
# P(X_t=0) = 0 -> P(X_t+1) = 1
# >>> a_em
# array([[  2.54354295e-293,   1.00000000e+000],
#        [  1.00000000e+000,   5.67486661e-014]])
# [0, 1;
# 1 ,0]
# >>> b_em
# array([[  5.00000000e-001,   5.00000000e-001,   1.49055507e-282],
#        [  0.00000000e+000,   5.00000000e-001,   5.00000000e-001]])
# [1/2, 1/2, 0;
#   0, 1/2, 1/2]
# >>> pi_em
# array([ 1.,  0.])

# the result says that the states always alternate
# the first state is always state 0
# ->
# P(X_t=0) = 0 -> P(X_t+1) = 1

# with 3 states 
(a_em, b_em, pi_em, logProbs)=hmm.runEM(array([0,1,1,2]),3,3)

# >>> a_em
# array([[  0.00000000e+00,   1.00000000e+00,   2.55929190e-27],
#        [  0.00000000e+00,   3.33333336e-01,   6.66666664e-01],
#        [  2.27221603e-10,   3.76441473e-10,   9.99999999e-01]])
# -> [0,1,0;
#     0,1/3,2/3;
#     0, 0, 1]
# >>> b_em
# array([[  1.00000000e+000,   0.00000000e+000,   9.03943361e-259],
#        [  0.00000000e+000,   1.00000000e+000,   1.48323304e-029],
#        [  0.00000000e+000,   3.33333331e-001,   6.66666669e-001]])
# -> [1,0,0; 0,1,0;, 0,1/3,2/3];
# pi_em
# array([ 1.,  0.,  0.])

# with 4 states 
(a_em, b_em, pi_em, logProbs)=hmm.runEM(array([0,1,1,2]),4,3)
# >>> a_em
# array([[ 0.        ,  0.        ,  0.        ,  1.        ],
#        [ 1.        ,  0.        ,  0.        ,  0.        ],
#        [ 0.        ,  1.        ,  0.        ,  0.        ],
#        [ 0.4312042 ,  0.07198119,  0.10470849,  0.39210612]])
# >>> b_em
# array([[ 0.,  1.,  0.],
#        [ 0.,  1.,  0.],
#        [ 1.,  0.,  0.],
#        [ 0.,  0.,  1.]])
# pi_em
# array([ 0.,  0.,  1.,  0.])

#### WU10
(words, wordsDict) = datasets.readCharacterFile("words.txt")
# >>> words
# array([ 8,  1, 22, ..., 12,  5, 19])
# >>> wordsDict[words]
# array(['h', 'a', 'v', ..., 'l', 'e', 's'], 
#       dtype='|S1')
(a_em, b_em, pi_em, logProbs)=hmm.runEM(words,2,27)

