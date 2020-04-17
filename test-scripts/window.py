import numpy as np
from matplotlib import pyplot as plt

from audioaugmentation.data import dms_to_numpy, window_examples

X_train, y_train, X_test, y_test = dms_to_numpy(1)

examples = [X_test[5], X_test[200]]
labels = np.vstack((y_test[5], y_test[200]))

plt.plot(examples[0])
print(labels[0])
plt.show()

plt.plot(examples[1])
print(labels[1])
plt.show()

examples_w, labels_w, index = window_examples(examples, labels, 32000, 0.5)

for ex, l, i in zip(examples_w, labels_w, index):
    plt.plot(ex)
    print(l)
    print(i)
    plt.show()
