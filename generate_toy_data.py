import numpy as np
import matplotlib.pyplot as plt
import csv

np.random.seed(0)

x = []
y = []
z = []
for i in np.linspace(-1, 1, 100):
    for j in np.linspace(-1, 1, 100):
        z.append(0.5 * np.sin(i) + np.tanh(3*j))
        x.append(i)
        y.append(j)

data = []
for i in range(500):
    a = np.random.uniform(0, len(x))
    data.append([x[int(a)], y[int(a)], z[int(a)]])

with open('toy_data.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(data)
csvFile.close()
