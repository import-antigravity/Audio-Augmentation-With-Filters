import tensorflow as tf
import os
from audioaugmentation.data import import_clean_data


def train(classifier, optimizer, epochs, batch_size, path, feature_size):
    print('Loading dataset')
    test_features, test_labels, train_features, train_labels = import_clean_data('../data/', feature_size)
    print(train_features.shape, train_labels.shape)
    classifier.compile(optimizer, loss=tf.keras.losses.mean_squared_logarithmic_error)
    print('Training...')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(min_lr=1e-8)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(path + '{epoch:02d}-{val_loss:.2f}.hdf5',
                                                    monitor='val_loss', verbose=0, save_best_only=True,
                                                    save_weights_only=False, mode='auto', period=1)
    csv_logger = tf.keras.callbacks.CSVLogger(path + "model_history_log.csv", append=True)
    history = classifier.fit(train_features, train_labels, validation_data=(test_features, test_labels),
                             batch_size=batch_size, epochs=epochs, callbacks=[checkpoint, csv_logger, reduce_lr])
    return classifier, optimizer, history
