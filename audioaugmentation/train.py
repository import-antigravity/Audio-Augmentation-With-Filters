import tensorflow as tf
from audioaugmentation.data import import_dataset


def train(classifier, optimizer, epochs, batch_size):
    print('Loading dataset')
    train, test = import_dataset()
    batched = train.batch(batch_size)
    classifier.compile(optimizer, loss=tf.keras.losses.binary_crossentropy)
    print('Training...')
    history = classifier.fit(batched, epochs=epochs, verbose=2)
    return classifier, optimizer, history, train, test
