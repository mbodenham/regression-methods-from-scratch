import csv
import numpy as np
from time import time
from split import split_data

def hypothesis(theta, X, n):
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    for i in range(0,X.shape[0]):
        h[i] = float(np.matmul(theta, X[i]))
    h = h.reshape(X.shape[0])
    return h

def gradient_descent(theta, alpha, num_iters, min_delta, h, X, y, n):
    cost = np.zeros(num_iters)

    for i in range(0,num_iters):
        theta[0] = theta[0] - (alpha/X.shape[0]) * sum(h - y)
        for j in range(1,n+1):
            theta[j] = theta[j] - \
                        (alpha/X.shape[0]) * sum((h-y) * X.transpose()[j])
        h = hypothesis(theta, X, n)
        cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))
        print('iteration: {}, cost: {}'.format(i, cost[i]))

        if cost[i-1] - cost[i] < min_delta and i != 0:
            print('Delta less then min_delta:', cost[i-1] - cost[i])
            break

    theta = theta.reshape(1,n+1)
    return theta, cost

def linear_regression(X, y, alpha, num_iters, min_delta):
    n = X.shape[1]
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1)
    theta = np.zeros(n+1)
    h = hypothesis(theta, X, n)
    theta, cost = gradient_descent(theta,alpha,num_iters,min_delta,h,X,y,n)
    return theta, cost

# x_train, y_train, x_test, y_test, y_test_s, y_train_s =\
#                             split_data('toy/toy_data.csv', scalling='standard')
x_train, y_train, x_test, y_test, y_test_s, y_train_s =\
                            split_data('sarcos_inv.csv', scalling='standard')

step = 0.49
n_iter = 100000
min_delta = 0.0000001
theta, cost = linear_regression(x_train, y_train, step, n_iter, min_delta)

print(theta)

np.savetxt('theta.csv', theta, delimiter=",")
np.savetxt('cost.csv', cost[np.nonzero(cost)], delimiter=",")
