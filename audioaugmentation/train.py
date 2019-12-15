import tensorflow as tf
from audioaugmentation.data import import_numpy


def train(classifier, optimizer, epochs, batch_size):
    print('Loading dataset')
    test_features, test_labels, train_features, train_labels = import_numpy('../data')
    classifier.compile(optimizer, loss=tf.keras.losses.binary_crossentropy)
    print('Training...')
    history = classifier.fit(train_features, train_labels, batch_size=batch_size, epochs=epochs, verbose=2)
    return classifier, optimizer, history
