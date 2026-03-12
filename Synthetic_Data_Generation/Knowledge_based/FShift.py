import numpy as np
from scipy.signal import hilbert

def nextpow2(x):
    return int(np.ceil(np.log2(np.abs(x))))

def freq_shift(x, f_shift, dt=1 / 250, num_channels=22):
    shifted_sig = np.zeros((x.shape))
    len_x = len(x)
    padding_len = 2 ** nextpow2(len_x)
    padding = np.zeros((padding_len - len_x, num_channels))
    with_padding = np.vstack((x, padding))
    hilb_T = hilbert(with_padding, axis=0)
    t = np.arange(0, padding_len)
    shift_func = np.exp(2j * np.pi * f_shift * dt * t)
    for i in range(num_channels):
        shifted_sig[:, i] = (hilb_T[:, i] * shift_func)[:len_x].real
    return shifted_sig


def freq_mod_f(data, labels, size, n_channels=22):
    new_data = []
    augsettings = 'pos'
    freq_mod = 0.2  # [0.1, 0.2, 0.3, 0.4, 0.5]
    data = np.swapaxes(data, -1, -2)
    print(data.shape)
    # print("freq mod: {}".format(freq_mod))
    for i in range(len(labels)):
        if labels[i] >= 0 and augsettings == 'neg':
            low_shift = freq_shift(data[i], -freq_mod, num_channels=n_channels)
            new_data.append(low_shift)
    for i in range(len(labels)):
        if labels[i] >= 0 and augsettings == 'pos':
            high_shift = freq_shift(data[i], freq_mod, num_channels=n_channels)
            new_data.append(high_shift)
    new_data_ar = np.array(new_data)
    new_data_ar = np.swapaxes(new_data_ar, -1, -2)
    return new_data_ar
