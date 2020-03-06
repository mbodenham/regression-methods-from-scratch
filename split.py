import numpy as np
import csv
import random

def normalise_data(data):
    return (data - data.min(0)) / data.ptp(0)

def standardise_data(data):
    return (data - data.mean(0)) / data.std(0)

def split_data(filename, split_ratio=0.8, n_values=False, scalling=None):
    with open(filename, 'r') as csv_file:
        data = list(csv.reader(csv_file))

    data = [[float(x) for x in lst] for lst in data]
    data = np.array(data)
    x = data[:, :-1]
    y = data[:, -1]

    np.random.seed(0)
    shuffle = np.random.permutation(data.shape[0])
    np.random.seed()

    if n_values:
        x = x[shuffle][:n_values]
        y = y[shuffle][:n_values]
    else:
        x = x[shuffle]
        y = y[shuffle]


    split = int(len(y) * split_ratio)
    x_train = x[:split]
    y_train = y[:split]
    x_test = x[split:]
    y_test = y[split:]

    if scalling == 'normal':
        x_test = normalise_data(x_test)
        x_train = normalise_data(x_train)
        y_test = normalise_data(y_test)
        y_train = normalise_data(y_train)

    if scalling == 'standard':
        x_test = standardise_data(x_test)
        x_train = standardise_data(x_train)

        y_test_mean = y_test.mean()
        y_test_std = y_test.std()
        y_train_mean = y_train.mean()
        y_train_std = y_train.std()
        y_test = (y_test - y_test_mean)/y_test_std
        y_train = (y_train- y_train_mean)/y_test_std
        y_test_s = (y_test_mean, y_test_std)
        y_train_s = (y_train_mean, y_train_std)

        return x_train, y_train, x_test, y_test, y_test_s, y_train_s

    return x_train, y_train, x_test, y_test

# x_train, y_train, x_test, y_test = split_data('sarcos_inv.csv', n_values=50)
