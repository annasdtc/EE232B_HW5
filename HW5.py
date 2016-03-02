#! /usr/bin/env python

# EE 232B - HW Set #5
# Anna Saez de Tejada Cuenca
# Due on March 2, 2016

# Load libraries
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

# Set random seed
rd.seed(1986)

# Length of the queues we want to generate
K = 10000

# Traffic intensities to try
R = np.arange(0.1,1.,0.1)
L = len(R)

# Vectors to store the statistics
MX1 = np.zeros(L)
MX2 = np.zeros(L)
MW1 = np.zeros(L)
MW2 = np.zeros(L)
SW1 = np.zeros(L)
SW2 = np.zeros(L)
PW1 = np.zeros(L)
PW2 = np.zeros(L)
RM = np.zeros(L)
RS = np.zeros(L)
RP = np.zeros(L)

for FIFO in [True,False]:

    out = open('./Results/'+str(FIFO)+'_Results.txt', 'w')

    for i in range(L):

        # Traffic intensity (rho)
        r = R[i]
        out.write('\n-----------------------------\n')
        out.write('\nTraffic intensity: '+str(r)+'\n')

        # Arrival rate (lambda)
        l = 9.0625/r
        out.write('Mean interarrival time:'+str(l)+'\n')

        # Generate L arrivals at rate lambda = 0.18955*rho
        I = rd.exponential(l,K)
        A = np.cumsum(I)

        # Randomly assign them class 2 with probability 1/16
        C = 1+(rd.rand(K)<1./16)

        # Generate L service times at rate 0.2 = 1/5 msec
        S = rd.exponential(5,K)

        # Change class 2 service times to 250 with probability 0.2, 25 with probability 0.8
        S[C==2] = 25 + 225*(rd.rand(1,np.sum(C==2))<0.2)

        # Vector to store departure times
        D = np.zeros(K)

        # First message departs with no queue at all
        D[0] = A[0]+S[0]

        # Simulate FIFO queue
        if FIFO:
            for k in range(K-1):
                D[k+1] = np.max((D[k],A[k+1]))+S[k+1]

        # Simulate priority queue
        else:
            k = 0
            K1 = np.asarray(np.nonzero(C==1))[0]
            K2 = np.asarray(np.nonzero(C==2))[0]
            k1 = 1*(C[0]==1)
            k2 = 1*(C[0]==2)
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
                elif (k1<len(K1) and k2<len(K2) and A[K1[k1]]<=A[K2[k2]]):
                    m = K1[k1]
                    D[m] = S[m] + A[m]
                    k1 = k1+1
                    k = m
                elif (k1<len(K1) and k2<len(K2) and A[K1[k1]]>A[K2[k2]]):
                    m = K2[k2]
                    D[m] = S[m] + A[m]
                    k2 = k2+1
                    k = m
                elif k1<len(K1):
                    m = K1[k1]
                    D[m] = S[m] + A[m]
                    k1 = k1+1
                    k = m
                else:
                    m = K2[k2]
                    D[m] = S[m] + A[m]
                    k2 = k2+1
                    k = m

        # Waiting times
        W = D-S-A
        W1 = W[C==1]
        W2 = W[C==2]

        # Vector of times in which there is an event and system sizes
        T = np.sort(np.concatenate((np.zeros(1),A,D)))
        X1 = np.zeros(2*K+1)
        X2 = np.zeros(2*K+1)
        for k in np.arange(1,2*K):
            if (sum(A==T[k])>0 or k==0):
                if C[A==T[k]]==1:
                    X1[k] = X1[k-1]+1
                    X2[k] = X2[k-1]
                else:
                    X1[k] = X1[k-1]
                    X2[k] = X2[k-1]+1
            else:
                if C[D==T[k]]==1:
                    X1[k] = X1[k-1]-1
                    X2[k] = X2[k-1]
                else:
                    X1[k] = X1[k-1]
                    X2[k] = X2[k-1]-1

        # Truncate the queue at the time of the last arrival to simulate steady-state
        tm = A[-1]
        X1 = X1[T<=tm]
        X2 = X2[T<=tm]
        T = T[T<=tm] 

        # Compute statistics
        mx1 = np.mean(X1)
        mx2 = np.mean(X2)
        mw1 = np.mean(W1)
        mw2 = np.mean(W2)
        sw1 = np.std(W1)
        sw2 = np.std(W2)
        pw1 = np.percentile(W1,95)
        pw2 = np.percentile(W2,95)
        rm = float(mw1)/float(mw2)
        rs = float(sw1)/float(sw2)
        rp = float(pw1)/float(pw2)

        # Store statistics
        MX1[i] = mx1
        MX2[i] = mx2
        MW1[i] = mw1
        MW2[i] = mw2
        SW1[i] = sw1
        SW2[i] = sw2
        PW1[i] = pw1
        PW2[i] = pw2
        RM[i] = rm
        RS[i] = rs
        RP[i] = rp

        # Print solutions
        out.write('a)\n')
        out.write('Mean system size of class 1 messages: '+str(mx1)+'\n')
        out.write('Mean system size of class 2 messages: '+str(mx2)+'\n')
        out.write('b)\n')
        out.write('Mean waiting time of class 1 messages: '+str(mw1)+'\n')
        out.write('Mean waiting time of class 2 messages: '+str(mw2)+'\n')
        out.write('c)\n')
        out.write('Standard devation of waiting time of class 1 messages: '+str(sw1)+'\n')
        out.write('Standard devation of waiting time of class 2 messages: '+str(sw2)+'\n')
        out.write('d)\n')
        out.write('Percentile 95 of waiting time of class 1 messages: '+str(pw1)+'\n')
        out.write('Percentile 95 of waiting time of class 2 messages: '+str(pw2)+'\n')
        out.write('e)\n')
        out.write('Ratio of mean waiting times: ' +str(rm)+'\n')
        out.write('Ratio of standard deviation of waiting times: ' +str(rs)+'\n')
        out.write('Ratio of percentile 95 of waiting times: ' +str(rp)+'\n')

        # Plot system sizes
        plt.clf()
        line1 = plt.plot(T,X1,'k-', label='Class 1')
        line2 = plt.plot(T,X2,'k--', label='Class 2')
        plt.title('Traffic intensity:'+str(r))
        plt.xlabel('Time')
        plt.ylabel('System size')
        plt.legend(loc=2)
        plt.savefig('./Results/'+str(FIFO)+'_'+str(r)+'_Size.pdf', bbox_inches='tight')

    # Plot evolution of the mean system size
    plt.clf()
    line1 = plt.plot(R,MX1,'k-', label='Class 1')
    line2 = plt.plot(R,MX2,'k--', label='Class 2')
    plt.title('Mean system size')
    plt.xlabel('Traffic intensity')
    plt.ylabel('Mean system size')
    plt.legend(loc=2)
    plt.savefig('./Results/'+str(FIFO)+'_MeanSize.pdf', bbox_inches='tight')

    # Plot evolution of the mean waiting time
    plt.clf()
    line1 = plt.plot(R,MW1,'k-', label='Class 1')
    line2 = plt.plot(R,MW2,'k--', label='Class 2')
    plt.title('Mean waiting time')
    plt.xlabel('Traffic intensity')
    plt.ylabel('Mean waiting time')
    plt.legend(loc=2)
    plt.savefig('./Results/'+str(FIFO)+'_MeanWait.pdf', bbox_inches='tight')

    # Plot evolution of the standard deviation of the waiting time
    plt.clf()
    line1 = plt.plot(R,SW1,'k-', label='Class 1')
    line2 = plt.plot(R,SW2,'k--', label='Class 2')
    plt.title('Standard deviation of the waiting time')
    plt.xlabel('Traffic intensity')
    plt.ylabel('Std of the waiting time')
    plt.legend(loc=2)
    plt.savefig('./Results/'+str(FIFO)+'_StdWait.pdf', bbox_inches='tight')

    # Plot evolution of the percentile of the waiting time
    plt.clf()
    line1 = plt.plot(R,PW1,'k-', label='Class 1')
    line2 = plt.plot(R,PW2,'k--', label='Class 2')
    plt.title('Percentile 95 of the waiting time')
    plt.xlabel('Traffic intensity')
    plt.ylabel('Percentile 95 of the waiting time')
    plt.legend(loc=2)
    plt.savefig('./Results/'+str(FIFO)+'_P95Wait.pdf', bbox_inches='tight')

    # Plot evolution of the ratio of mean waiting times
    plt.clf()
    plt.plot(R,RM,'k-')
    plt.title('Ratio of mean waiting times')
    plt.xlabel('Traffic intensity')
    plt.ylabel('Ratio of mean waiting times')
    plt.savefig('./Results/'+str(FIFO)+'_RatioMeanWait.pdf', bbox_inches='tight')

    # Plot evolution of the ratio of mean waiting times
    plt.clf()
    plt.plot(R,RS,'k-')
    plt.title('Ratio of standard deviation of waiting times')
    plt.xlabel('Traffic intensity')
    plt.ylabel('Ratio of std of waiting times')
    plt.savefig('./Results/'+str(FIFO)+'_RatioStdWait.pdf', bbox_inches='tight')

    # Plot evolution of the ratio of mean waiting times
    plt.clf()
    plt.plot(R,RP,'k-')
    plt.title('Ratio of percentile 95 of waiting times')
    plt.xlabel('Traffic intensity')
    plt.ylabel('Ratio of percentile 95 of waiting times')
    plt.savefig('./Results/'+str(FIFO)+'_RatioP95Wait.pdf', bbox_inches='tight')

    # Close file
    out.close()
