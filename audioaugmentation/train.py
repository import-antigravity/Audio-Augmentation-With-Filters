import tensorflow as tf
from audioaugmentation.data import import_numpy


def train(classifier, optimizer, epochs, batch_size):
    print('Loading dataset')
    test_features, test_labels, train_features, train_labels = import_numpy('../data')
    print(train_features.shape, train_labels.shape)
    classifier.compile(optimizer, loss=tf.keras.losses.binary_crossentropy)
    print('Training...')
    history = classifier.fit(train_features, train_labels, batch_size=batch_size, epochs=epochs)
    return classifier, optimizer, history
