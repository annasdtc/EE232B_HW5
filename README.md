# EE232B_HW5
UCLA EE 232B - HW Set #5 - M/G/1 queuing system, FIFO/priority type queues

* Simulation of a queuing system with two classes of messages
* Class 1 arrive at a rate lambda(1) and have service time exponentially distributed with mean 5 msec.
* Class 2 arrive at a rate lambda(2) and have fixed service time 25 msec with probability 0.8 or 250 msec with probability 0.2
* lambda(1) = 15*lambda(2)
* Traffic intensity rho from 0.1 to 0.9 in increments of 0.1
* Queue is either FIFO or non-preemptive priority (class 1 are high priority)

Compute:
a. Mean system size for each class
b. Average waiting time for each class
c. Standard deviation of the waiting time for each class
d. Percentile 95 of waiting time for each class
e. Ratio of class 2 to class 1 for means, standard deviations and percentile 95
f. Maximum loading rate to assure average delays are at most 10 msec for class 1 and 150 msec for class 2 (priority queue case)
