import tensorflow as tf
from audioaugmentation.data import import_numpy


def train(classifier, optimizer, epochs, batch_size):
    print('Loading dataset')
    test_features, test_labels, train_features, train_labels = import_numpy('../data')
    print(train_features.shape, train_labels.shape)
    classifier.compile(optimizer, loss=tf.keras.losses.binary_crossentropy)
    print('Training...')
    checkpoint = tf.keras.callbacks.ModelCheckpoint('.', monitor='val_loss', verbose=0, save_best_only=True,
                                                    save_weights_only=False, mode='auto', period=1)
    csv_logger = tf.keras.callbacks.CSVLogger("model_history_log.csv", append=True)
    history = classifier.fit(train_features, train_labels, validation_data=(test_features, test_labels),
                             batch_size=batch_size, epochs=epochs, callbacks=[checkpoint, csv_logger])
    return classifier, optimizer, history
