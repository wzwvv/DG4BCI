import numpy as np

def data_noise_f(data, labels, size, n_channels=22):
    new_data = []
    noise_mod_val = 2   # [0.25, 0.5, 1, 2, 4]
    for i in range(len(labels)):
        if labels[i] >= 0:
            stddev_t = np.std(data[i])
            rand_t = np.random.rand(data[i].shape[0], data[i].shape[1])
            rand_t = rand_t - 0.5
            to_add_t = rand_t * stddev_t / noise_mod_val
            data_t = data[i] + to_add_t
            new_data.append(data_t)
    new_data_ar = np.array(new_data)
    return new_data_ar