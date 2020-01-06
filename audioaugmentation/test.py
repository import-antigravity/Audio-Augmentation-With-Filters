import numpy as np
from matplotlib import pyplot as plt

from audioaugmentation.models import cnn_rand

print("Loading models")
o = cnn_rand()
o.load_weights('../models/clean/32-0.02.hdf5')
print(o.summary())

a = cnn_rand()
a.load_weights('../hpc-scripts/aug/30-0.03.hdf5')
print(a.summary())

labels = ['air conditioner', 'car horn', 'dog bark', 'children playing', 'drilling', 'engine idling', 'gun shot',
          'jackhammer', 'siren', 'street music']

print("Loading data")
npz = np.load('../data/augmented_val.npz')
f_n, l_n = npz.files
test_features, test_labels = npz[f_n], npz[l_n]

print("Predicting")
predict_o = np.argmax(o.predict(test_features), axis=1)
predict_a = np.argmax(a.predict(test_features), axis=1)
truth = np.argmax(test_labels, axis=1)

accuracy_o = [np.logical_and(predict_o == i, truth == i).sum() / (predict_o == i).sum() for i in range(10)]
print(accuracy_o)
accuracy_a = [np.logical_and(predict_a == i, truth == i).sum() / (predict_a == i).sum() for i in range(10)]
print(accuracy_a)

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(16, 4))
rects1 = ax.bar(x - width / 2, accuracy_o, width, label='Control')
rects2 = ax.bar(x + width / 2, accuracy_a, width, label='Augmented')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Evaluated on Dataset with Simulated IRs')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.savefig('aug_bar.png')
