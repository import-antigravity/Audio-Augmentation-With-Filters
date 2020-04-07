import os
from typing import Tuple

import numpy as np
from tensorflow import keras


def train_baseline(model, optimizer, loss, name: str, num_epochs: int, batch_size: int,
                   data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], callbacks: list = None,
                   model_path: str = None):
    if model_path is None:
        model_path = f'../models/{name}'
    if callbacks is None:
        callbacks = [
            keras.callbacks.ModelCheckpoint(os.path.join(model_path, '{epoch:02d}-{val_loss:.2f}.hdf5'),
                                            monitor='val_loss', verbose=0, save_best_only=True,
                                            save_weights_only=True, mode='auto', period=1),
            keras.callbacks.CSVLogger(os.path.join(model_path, "model_history_log.csv"), append=True)
        ]

    print(model.summary())

    X_train, y_train, X_test, y_test = data

    os.makedirs(model_path, exist_ok=True)

    print('Dataset:', X_train.shape, y_train.shape)
    model.compile(optimizer, loss=loss)
    print('Training...')
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=num_epochs,
                        callbacks=callbacks, shuffle=True)
    return model, optimizer, history
