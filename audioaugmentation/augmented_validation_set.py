import random

import numpy as np

from audioaugmentation.data import data_frame_to_folds, combine_folds, room_distribution, conform_examples

data_folds, label_folds = data_frame_to_folds('../data')
test_features, test_labels, train_features, train_labels = combine_folds(data_folds, label_folds)

print(len(test_features))

# Augment
num_rooms = 300
rd = room_distribution(num_rooms)
irs = rd.make_irs(num_rooms)

print("Applying augmentation:")
# Generate IRs
print("    Generating IRs", end='')
ir_for_example = []
j = 0
for _ in test_features:
    ir_for_example.append(random.choice(irs))
    j += 1
    if j % 50 == 0:
        print('.', end='')
print()

# Do convolution
print('    Convolving', end='')
augmented = []
j = 0
for x, ir in zip(test_features, ir_for_example):
    augmented.append(np.convolve(x, ir))
    j += 1
    if j % 50 == 0:
        print('.', end='')
print()

test_features, test_labels = conform_examples(augmented, test_labels, 32000, 0.5)

np.savez('augmented_val', test_features, test_labels)
