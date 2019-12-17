import os

import tensorflow as tf


def train(data, classifier, optimizer, epochs, batch_size, model_path, num_gpus):
    os.makedirs(model_path)
    print('Loading dataset')
    test_features, test_labels, train_features, train_labels = data
    print('Dataset:', train_features.shape, train_labels.shape)
    if num_gpus > 1:
        model = tf.keras.utils.multi_gpu_model(classifier, num_gpus)
    else:
        model = classifier
    model.compile(optimizer, loss=tf.keras.losses.mean_squared_logarithmic_error)
    print('Training...')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(min_lr=1e-8)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_path, '{epoch:02d}-{val_loss:.2f}.hdf5'),
                                                    monitor='val_loss', verbose=0, save_best_only=True,
                                                    save_weights_only=False, mode='auto', period=1)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_path, "model_history_log.csv"), append=True)
    # Train model
    history = model.fit(train_features, train_labels, validation_data=(test_features, test_labels),
                        batch_size=batch_size, epochs=epochs, callbacks=[checkpoint, csv_logger, reduce_lr])
    return model, optimizer, history
