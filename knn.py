import csv
import multiprocessing as mp
import numpy as np
import random
from joblib import Parallel, delayed
from math import sqrt
from split import split_data

def euclidean_distance(p, q):
    return sqrt(sum((p - q)**2))

def find_neighbours(train, test_row, k):
    distances = []
    for i, r in enumerate(train):
        dist = euclidean_distance(test_row, r)
        distances.append((dist, i))
    distances = sorted(distances)
    return distances[:k]

def get_rsme(x_train, x_test, y_train, y_test, k, sample):
    RSME = []
    MAE = []
    for i, test_row in enumerate(x_test[:sample]):
        estimations = []
        for n in find_neighbours(x_train, test_row, k)[1:]:
            estimations.append(y_train[n[1]])
        actual = y_test[i]
        estimate = np.mean(estimations)
        RSME.append((estimate - actual)**2)
        MAE.append(abs(estimate - actual))
    RSME = sqrt(sum(RSME)/len(RSME))
    MAE = sum(MAE)/len(MAE)
    print('{} - MAE:{}, RSME: {},'.format(k, MAE, RSME))
    return [k, MAE, RSME]

x_train, y_train, x_test, y_test = split_data('data/toy_data.csv', scalling='normal')
# x_train, y_train, x_test, y_test = split_data('sarcos_inv.csv', sampling='normal')

## Random test
# RSME = []
# MAE = []
# for i, actual in enumerate(y_test):
#     print(i, len(y_test))
#     estimation = random.uniform(min(y_train), max(y_train))
#     RSME.append((estimation - actual)**2)
#     MAE.append(abs(estimation -actual))
# RSME = sqrt(sum(RSME)/len(RSME))
# MAE = sum(MAE)/len(MAE)
# print('MAE: {}, RSME: {}'.format(MAE, RSME))

sample = 10000
k_max = 50
cores = mp.cpu_count()
results = Parallel(n_jobs=cores)(delayed(get_rsme)(x_train, x_test, y_train,\
                                y_test, i, sample) for i in range(2, k_max))

np.savetxt('data/knn_toy.csv', results, delimiter=",")

best_k = (0, 1000)
for k in results:
    if k[1]+k[2] < best_k[1]:
        best_k = (k[0], k[1]+k[2])


k = best_k[0]
print()
print('\nBest k:', k)
get_rsme(x_train, x_test, y_train, y_test, k, sample)
