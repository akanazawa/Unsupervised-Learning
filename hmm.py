from numpy import *
from pylab import *
from util  import *

def viterbi(X, a, b, pi):
    """
    here we implement the viterbi algorithm

    a,b,pi are the model parameters.  that is:
      a[i,j] = p(state_t = j | state_{t-1} = i)
      b[i,x] = p(observation = x | state = i)
      pi[i]  = p(state_0 = i)

    X is an array of observations.  
    
    you should produce a matrix Z that looks exactly like X, but is
    state sequences rather than observation sequences.
    """

    K,V = a.shape
    # K is the number of states, V is the number of unique observations

    T = X.shape[0]

    # run the viterbi algorithm on sequence X

    # al[k,t] stores the *log* probability that the most likely
    # path through state k at time t; ze[k,t] stores the
    # approrpiate back pointer
    al = zeros((K,T+1), dtype=float)
    ze = zeros((K,T+1), dtype=int)

    # initialize for t=0
    for k in range(K):
        al[k,0] = log(pi[k])
        ze[k,0] = -1

    # loop for time
    for t in range(1,T+1):
        for k in range(K):
            ### TODO: YOUR CODE HERE
            util.raiseNotDefined()
            al[k,t] = util.raiseNotDefined()    ### TODO: YOUR CODE HERE

            ze[k,t] = util.raiseNotDefined()    ### TODO: YOUR CODE HERE


    print al
    print ze

    # back track best path
    path = zeros((T,), dtype=int) - 1

    # initialize at end
    path[T-1] = util.raiseNotDefined()    ### TODO: YOUR CODE HERE


    # recurse backward
    for t in range(T-1,0,-1):
        path[t-1] = util.raiseNotDefined()    ### TODO: YOUR CODE HERE


    return path

def initializeHMMRandomly(K,V):
    """
    initialize pi, a and b randomly
    """

    pi = rand(K)      # initialize randomly
    a  = rand(K,K)
    b  = rand(K,V)

    pi = pi / sum(pi) # normalize
    for k in range(K):
        a[k,:] = a[k,:] / sum(a[k,:])
        b[k,:] = b[k,:] / sum(b[k,:])

    return (a,b,pi)

def forward(X, a, b, pi):
    """
    this is exactly like viterbi, except we return a lattice al of
    forward probabilities, rather than optimal state sequences:

    hint: copy and paste is your friend!
    """

    K,V = a.shape
    # K is the number of states, V is the number of unique observations

    T = X.shape[0]

    # al[k,t] stores the *log* probability that paths go through
    # state k at time t
    al = zeros((K,T+1), dtype=float)

    # remember: al[k,t] includes transitions *to* state k at time
    # t, and observations up to *but not including* X[t]

    # initialize for t=0
    for k in range(K):
        al[k,0] = util.raiseNotDefined()    ### TODO: YOUR CODE HERE


    # loop for time
    for t in range(1,T+1):
        for k in range(K):
            val = -inf
            ### TODO: YOUR CODE HERE
            util.raiseNotDefined()
            al[k,t] = val

    return al

def backward(X, a, b, pi):
    """
    exactly like forward, but return BE array of backward probabilities
    """

    K,V = a.shape
    # K is the number of states, V is the number of unique observations

    T = X.shape[0]

    # run the viterbi algorithm on sequence X

    # be[k,t] stores the *log* probability that paths go through
    # state k at time t
    be = zeros((K,T+1), dtype=float)

    # remember: be[k,t] includes transitions *from* state k at time
    # t, and observations from *and including* X[t]

    # initialize for t=0
    for k in range(K):
        be[k,T] = 0  # log(1) = 0

    # loop for time
    for t in range(T-1, -1, -1):
        for k in range(K):
            val = -inf
            ### TODO: YOUR CODE HERE
            util.raiseNotDefined()
            be[k,t] = val

    return be

def reestimate(X, al, be, aold, bold, piold):
    """
    re-estimate model probabilities pi, a and b on the
    basis of the forward-backward probabilities AL and BE
    on data X.  aold is the old values of a and bold is 
    the old values of b.
    """

    K,V = bold.shape

    pi = zeros((K,)) - inf
    a  = zeros((K,K)) - inf
    b  = zeros((K,V)) - inf

    # compute the log probability of the entire sequence
    logpXn = -inf
    for k in range(K):
        logpXn = addLog(logpXn, al[k,-1])

    T  = X.shape[0]

    # re-estimate start states (pi):
    for k in range(K):
        ### TODO: YOUR CODE HERE
        util.raiseNotDefined()

    # re-estimate emissions (b):
    for k in range(K):
        ### TODO: YOUR CODE HERE
        util.raiseNotDefined()

    # re-estimate transitions (a):
    for k in range(K):
        for k_next in range(K):
            ### TODO: YOUR CODE HERE
            util.raiseNotDefined()

    # normalize the new probabilities
    pi = normalizeLog(pi)
    for k in range(K):
        a[k,:] = normalizeLog(a[k,:])
        b[k,:] = normalizeLog(b[k,:])

    return (a,b,pi)

def dataLogProbability(al):
    logpXn = -inf
    for k in range(al.shape[0]):
        logpXn = addLog(logpXn, al[k,-1])
    return logpXn

def sanityCheck(al, be):
    K,T = al.shape
    pX  = zeros((T,))
    for t in range(T):
        v = -inf
        for k in range(K):
            v = addLog(v, al[k,t] + be[k,t])
        pX[t] = v
    if max(pX) - min(pX) > 1e-6:
        print "Warning: sanity check failed!"
        print pX


def printHMMModel(a, b, pi):
    K,V = b.shape

    print "Initial state probabilities:"
    for k in range(K):
        print "\tstate(%d):\t%g" % (k, pi[k])

    print ""
    print "Transition probabilities:"
    print "FROM\\TO"
    for k in range(K):
        print "\t%d" % k,
    print ""
    for k in range(K):
        print "%d" % k,
        for k_next in range(K):
            print "\t%g" % a[k,k_next],
        print ""

    print ""
    print "Emission probabilities:"
    for k in range(K):
        print "\tState %d:" % k
        ids = argsort(-b[k,:])
        for v in range(V):
            s = "_"
            if ids[v] > 0:
                s = chr(ids[v] + ord('a') - 1)
            print "\t\t%s  %g" % (s, b[k,ids[v]])

def printViterbi(X, Z):
    T = X.shape[0]
    for t in range(T):
        s = "_"
        if X[t] > 0:
            s = chr(X[t] + ord('a') - 1)
        print "%s" % s,
    print ""
    for t in range(T):
        print "%d" % Z[t],
    print ""

def runEM(X, K, V):
    """
    run EM on the HMM model
    """

    (a,b,pi) = initializeHMMRandomly(K,V)
    logProbs = []

    for iter in range(50):
        al = forward( X, a, b, pi)
        be = backward(X, a, b, pi)
        sanityCheck(al, be)

        logProb = dataLogProbability(al)
        logProbs.append(logProb)

        print "iteration %d\t... log probability %g" % (iter+1, logProb)

        aold  = a.copy()
        bold  = b.copy()
        piold = pi.copy()

        (a,b,pi) = reestimate(X, al, be, aold, bold, piold)

    if V==27:
        printHMMModel(a,b,pi)

    return (a,b,pi,logProbs)

