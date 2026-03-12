import numpy as np

def data_neg_f(data, labels, size, n_channels=22):
    # Returns: data double the size of the input over time, with new data
    # being a reflection along the amplitude
    new_data = []
    for i in range(len(labels)):
        if labels[i] >= 0:
            data_t = -1 * data[i]
            data_t = data_t - np.min(data_t)
            new_data.append(data_t)
    new_data_ar = np.array(new_data)
    return new_data_ar
