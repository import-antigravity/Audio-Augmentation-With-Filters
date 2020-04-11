import numpy as np
from matplotlib import pyplot as plt

from audioaugmentation.data import dms_to_numpy
from audioaugmentation.train import DataGenerator

dg = DataGenerator(np.zeros(1), np.zeros(1), True, False, 10, 32000, 0.5)
filters = dg.make_filters(10, 16)

X = np.vstack((dms_to_numpy(1)[0][0].reshape(1, -1),) * 10)
signals = dg.filter(X)

fig, ax = plt.subplots(4, 4)

for i, a in enumerate([j for i in ax for j in i]):
    a.plot(filters[i])
plt.show()

for i in range(10):
    plt.plot(signals[i], linewidth=0.1)
plt.show()
