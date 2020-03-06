import numpy as np
import csv

np.random.seed(0)

data = []
for _ in range(500):
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    z = 0.5 * np.sin(x) + np.tanh(3*y)

    data.append([x, y, z])

with open('toy_data.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(data)
csvFile.close()
