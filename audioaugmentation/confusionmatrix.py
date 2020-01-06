import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn import metrics

from audioaugmentation.data import data_frame_to_folds, combine_folds, conform_examples

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from audioaugmentation.models import cnn_rand

print("loading model")
model_au = cnn_rand()
model_au.load_weights('../models/clean/32-0.02.hdf5')
print(model_au.summary())

print("loading data")

data_folds, label_folds = data_frame_to_folds('../data')
test_features, test_labels, _, _ = combine_folds(data_folds, label_folds)
test_features, test_labels = conform_examples(test_features, test_labels, 32000, 0.5)

print("Predicting")
predictedvalue_au = model_au.predict(test_features)

print("Plotting")

cm_au = metrics.confusion_matrix(y_true=np.argmax(test_labels, 1), y_pred=np.argmax(predictedvalue_au, 1))
plt.figure(figsize=(12, 8))
sn.set(font_scale=0.8)  # for label size
x_axis_labels = ['air conditioner', 'car horn', 'dog bark', 'children playing', 'drilling', 'engine idling', 'gun shot',
                 'jackhammer', 'siren', 'street music']

sn.heatmap(cm_au, annot=True, annot_kws={"size": 10}, cmap="Blues", fmt='g', xticklabels=x_axis_labels,
           yticklabels=x_axis_labels)  # font size
plt.savefig('aug_clean.png')
