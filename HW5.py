#! /usr/bin/env python

# EE 232B - HW Set #5
# Anna Saez de Tejada Cuenca
# Due on March 2, 2016

# Load libraries
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

# Set random seed
#rd.seed(1986)

# Length of the queue we want to generate
K = 100

# Kind of queue: F = FIFO
F = False

# Traffic intensity (rho)
r = 0.1

# Generate L arrivals at rate lambda = 0.18955*rho
I = rd.exponential(0.18955*r,K)
A = np.cumsum(I)

# Randomly assign them class 2 with probability 1/16
T = 1+(rd.rand(K)<1./16)

# Generate L service times at rate 0.2 = 1/5 msec
S = rd.exponential(0.2,K)

# Change class 2 service times to 250 with probability 0.2, 25 with probability 0.8
S[T==2] = 25 + 225*(rd.rand(1,np.sum(T==2))<0.2)

# Vector to store departure times
D = np.zeros(K)

# First message departs with no queue at all
D[0] = A[0]+S[0]

# Simulate FIFO queue
if F:
    for k in range(K-1):
        D[k+1] = D[k]+S[k+1]

# Simulate priority queue
else:
    k = 0
    K1 = np.asarray(np.nonzero(T==1))[0]
    K2 = np.asarray(np.nonzero(T==2))[0]
    k1 = 1*(T[0]==1)
    k2 = 1*(T[0]==2)
    while k1+k2<K:
        if (k1<len(K1) and A[K1[k1]]<=D[k]):
            m = K1[k1]
            D[m] = S[m] + D[k]
            k1 = k1+1
            k = m
        elif (k2<len(K2) and A[K2[k2]]<=D[k]):
            m = K2[k2]
            D[m] = S[m] + D[k]
            k2 = k2+1
            k = m
        elif A[K1[k1]]<=A[K2[k2]]:
            m = K1[k1]
            D[m] = S[m] + A[m]
            k1 = k1+1
            k = m
        else:
            m = K2[k2]
            D[m] = S[m] + A[m]
            k2 = k2+1
            k = m

# Vector of times in which there is an event and system sizes
T = np.sort(np.concatenate((A,D)))
X = np.zeros(2*K)
for k in range(2*K-1):
    if sum(np.nonzero(A==T[k]))>0:
        X[k+1] = X[k]+1
    else:
        X[k+1] = X[k]-1

# Plot the system size in a step-wise way
t = np.zeros(4*K)
for k in range(2*K-1):
    t[2*k+1] = T[k]
    t[2*k] = t[2*k+1] - 1e-9
x = np.zeros(4*K)
for k in range(2*K-1):
    x[2*k+1] = X[k]
    x[2*k] = 1*(k>1)*X[k-1]
plt.plot(t,x)
plt.show()
