import numpy as np

from audioaugmentation.data import import_augmented_data


def yeet():
    test_features, test_labels, train_features, train_labels = import_augmented_data(
        data_path='../data',
        feature_size=32000,
        augmentation_factor=2,
        noise_mean=0,
        noise_stddev=0,
        num_rooms=300,
    )
    np.savez('augmented', test_features, test_labels, train_features, train_labels)


yeet()
