import numpy as np


def conform_examples(X_list: [np.ndarray], y_original: np.ndarray, window_size: int, crossover: float):
    assert len(X_list) == y_original.size

    X = []
    y = []

    for i in range(len(X_list)):
        sample = X_list[i]
        if sample.size <= window_size:
            zeros = np.zeros(window_size - sample.size)
            padded = np.concatenate((sample, zeros))
            X.append(padded)
            y.append(y_original[i])
        else:
            current_start = 0
            while current_start < sample.size:
                zeros = np.array([])
                if current_start + window_size > sample.size:
                    zeros = np.zeros(current_start + window_size - sample.size)
                padded = np.concatenate((sample[current_start:current_start + window_size], zeros))
                X.append(padded)
                y.append(y_original[i])
                current_start += int((1. - crossover) * window_size)

    return np.vstack(X), np.vstack(y)
