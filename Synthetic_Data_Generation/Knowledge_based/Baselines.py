# -*- coding: utf-8 -*-
# @Time    : 2021/6/25 21:01
# @Author  : wenzhang
# @File    : data_augment.py

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
from pyhht import EMD

def data_aug(data, labels, size, flag_aug):
    # augments data based on boolean inputs reuse_data, noise_data, neg_data, freq_mod data.
    # data: samples * size * n_channels
    # size: int(freq * window_size)
    # Returns: entire training dataset after data augmentation, and the corresponding labels

    # noise_flag, neg_flag, mult_flag, freq_mod_flag test 75.154
    # mult_flag, noise_flag, neg_flag, freq_mod_flag test 76.235
    # noise_flag, neg_flag, freq_mod_flag test 76.157

    mult_flag, noise_flag, neg_flag, freq_mod_flag = flag_aug[0], flag_aug[1], flag_aug[2], flag_aug[3]

    n_channels = data.shape[2]
    data_out = data  # 1 raw features
    labels_out = labels

    if mult_flag:  # 2 features
        mult_data_add, labels_mult = data_mult_f(data, labels, size, n_channels=n_channels)
        data_out = mult_data_add
    if noise_flag:  # 1 features
        noise_data_add, labels_noise = data_noise_f(data, labels, size, n_channels=n_channels)
        data_out = noise_data_add
    if neg_flag:  # 1 features
        neg_data_add, labels_neg = data_neg_f(data, labels, size, n_channels=n_channels)
        data_out = neg_data_add
    if freq_mod_flag:  # 2 features
        freq_data_add, labels_freq = freq_mod_f(data, labels, size, n_channels=n_channels)
        data_out = freq_data_add

    # Final output data layout
    # raw 144, mult_add 144, mult_reduce 144, noise 144, neg 144, freq1 144, freq2 144
    return data_out, labels_out


def data_noise_f(data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    noise_mod_val = 2
    # print("noise mod: {}".format(noise_mod_val))
    for i in range(len(labels)):
        if labels[i] >= 0:
            stddev_t = np.std(data[i])
            rand_t = np.random.rand(data[i].shape[0], data[i].shape[1])
            rand_t = rand_t - 0.5
            to_add_t = rand_t * stddev_t / noise_mod_val
            data_t = data[i] + to_add_t
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


def data_mult_f(data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    mult_mod = 0.1
    # print("mult mod: {}".format(mult_mod))
    # for i in range(len(labels)):
    #     if labels[i] >= 0:
    #         # print(data[i])
    #         data_t = data[i] * (1 + mult_mod)
    #         new_data.append(data_t)
    #         new_labels.append(labels[i])

    for i in range(len(labels)):
        if labels[i] >= 0:
            data_t = data[i] * (1 - mult_mod)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


def data_neg_f(data, labels, size, n_channels=22):
    # Returns: data double the size of the input over time, with new data
    # being a reflection along the amplitude

    new_data = []
    new_labels = []
    for i in range(len(labels)):
        if labels[i] >= 0:
            data_t = -1 * data[i]
            data_t = data_t - np.min(data_t)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


def freq_mod_f(data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    # print(data.shape)
    freq_mod = 0.2
    # print("freq mod: {}".format(freq_mod))
    for i in range(len(labels)):
        if labels[i] >= 0:
            low_shift = freq_shift(data[i], -freq_mod, num_channels=n_channels)
            new_data.append(low_shift)
            new_labels.append(labels[i])

    for i in range(len(labels)):
        if labels[i] >= 0:
            high_shift = freq_shift(data[i], freq_mod, num_channels=n_channels)
            new_data.append(high_shift)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


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


def nextpow2(x):
    return int(np.ceil(np.log2(np.abs(x))))


def HHTAnalysis(eegRaw, fs):
    # EMD decomposition
    decomposer = EMD(eegRaw)
    # Get IMF components after EMD
    imfs = decomposer.decompose()
    # Number of components after decomposition
    n_components = imfs.shape[0]
    # Figure: raw data and each component
    fig, axes = plt.subplots(n_components + 1, 2, figsize=(10, 7), sharex=True, sharey=False)
    # Plot raw data
    axes[0][0].plot(eegRaw)
    # Hilbert transform of raw data
    eegRawHT = hilbert(eegRaw)
    # Plot Hilbert transform result
    axes[0][0].plot(abs(eegRawHT))
    # Set plot title
    axes[0][0].set_title('Raw Data')
    # Instantaneous frequency after Hilbert transform
    instf, timestamps = tftb.processing.inst_freq(eegRawHT)
    # Plot instantaneous frequency; multiply by fs to convert to Hz
    axes[0][1].plot(timestamps, instf * fs)
    # Mean and median of instantaneous frequency
    axes[0][1].set_title('Freq_Mean{:.2f}----Freq_Median{:.2f}'.format(np.mean(instf * fs), np.median(instf * fs)))

    # Compute and plot each component
    for iter in range(n_components):
        # Plot IMF component
        axes[iter + 1][0].plot(imfs[iter])
        # Hilbert transform of each component
        imfsHT = hilbert(imfs[iter])
        # Plot Hilbert transform of component
        axes[iter + 1][0].plot(abs(imfsHT))
        # Set subplot title
        axes[iter + 1][0].set_title('IMF{}'.format(iter))
        # Instantaneous frequency of component after Hilbert transform
        instf, timestamps = tftb.processing.inst_freq(imfsHT)
        # Plot instantaneous frequency; multiply by fs to convert to Hz
        axes[iter + 1][1].plot(timestamps, instf * fs)
        # Mean and median of instantaneous frequency
        axes[iter + 1][1].set_title(
            'Freq_Mean{:.2f}----Freq_Median{:.2f}'.format(np.mean(instf * fs), np.median(instf * fs)))
    plt.tight_layout()
    plt.show()


# HHT filter: extract (a subset of) EMD components
def HHTFilter(eegRaw):
    # EMD decomposition
    decomposer = EMD(eegRaw)
    # Get IMF components after EMD
    imfs = decomposer.decompose()
    return imfs
