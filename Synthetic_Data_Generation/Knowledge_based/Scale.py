import numpy as np


def data_mult_f(data, labels, size, n_channels=22):
    new_data = []
    mult_mod = 0.05
    for i in range(len(labels)):
        if labels[i] >= 0:
            data_t = data[i] * (1 - mult_mod)
            new_data.append(data_t)
    new_data_ar = np.array(new_data)
    return new_data_ar