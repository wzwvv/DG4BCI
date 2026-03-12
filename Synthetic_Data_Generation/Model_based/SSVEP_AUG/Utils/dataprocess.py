from scipy import signal
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from etc.global_config import config
from Utils.Trainer import train_aug_generator
from pandas.io.sas.sas_constants import dataset_length
from types import SimpleNamespace
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import os
from scipy.fftpack import fft, dct, idct

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
import pywt
from pyhht import EMD


# 9 model-based augmentation types: 3 architectures x 3 loss variants
MODEL_BASED_AUGS = [
    'CNN-base', 'CNN-gan', 'CNN-vae',
    'CNN_Trans-base', 'CNN_Trans-gan', 'CNN_Trans-vae',
    'CNN_LSTM-base', 'CNN_LSTM-gan', 'CNN_LSTM-vae'
]

def fix_random_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for multi-GPU

    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # cuDNN settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def data_aug(data,labels,augtype,info=None):

    # augments data
    # data: samples * size * n_channels
    # size: int(freq * window_size)
    # Returns: entire training dataset after data augmentation, and the corresponding labels
    data =np.array(data.squeeze().swapaxes(1, 2))
    labels = np.array(labels)
    batch = data.shape[0]
    size = data.shape[1]
    n_channels = data.shape[2]
    data_out = data  # 1 raw features
    labels_out = labels
    aug_method = augtype
    # if mult_flag:  # 2 features
    #     mult_data_add, labels_mult = data_mult_f(data, labels, size, n_channels=n_channels)
    #     data_out = np.concatenate([data_out, mult_data_add], axis=0)
    #     labels_out = np.append(labels_out, labels_mult)
    # if noise_flag:  # 1 features
    #     noise_data_add, labels_noise = data_noise_f(data, labels, size, n_channels=n_channels)
    #     data_out = np.concatenate([data_out, noise_data_add], axis=0)
    #     labels_out = np.append(labels_out, labels_noise)
    # if neg_flag:  # 1 features
    #     neg_data_add, labels_neg = data_neg_f(data, labels, size, n_channels=n_channels)
    #     data_out = np.concatenate([data_out, neg_data_add], axis=0)
    #     labels_out = np.append(labels_out, labels_neg)
    # if freq_mod_flag:  # 2 features
    #     freq_data_add, labels_freq = freq_mod_f(data, labels, size, n_channels=n_channels)
    #     data_out = np.concatenate([data_out, freq_data_add], axis=0)
    #     labels_out = np.append(labels_out, labels_freq)
    if aug_method == 'Scale': 
        mult_data_add, labels_mult = data_mult_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, mult_data_add], axis=0)
        labels_out = np.append(labels_out, labels_mult)
    if aug_method == 'Noise':
        noise_data_add, labels_noise = data_noise_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, noise_data_add], axis=0)
        labels_out = np.append(labels_out, labels_noise)
    if aug_method == 'Flip': 
        neg_data_add, labels_neg = data_neg_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, neg_data_add], axis=0)
        labels_out = np.append(labels_out, labels_neg)
    if aug_method == 'FShift': 
        freq_data_add, labels_freq = freq_mod_f(data, labels, size, n_channels=n_channels)
        data_out = np.concatenate([data_out, freq_data_add], axis=0)
        labels_out = np.append(labels_out, labels_freq)
    if aug_method == 'DWTaug': 
        data_out=data_out.swapaxes(1, 2)
        data_out , labels_out = DWTaug_multi(data_out,labels_out)
        data_out=data_out.swapaxes(1, 2)
    if aug_method == 'HHTaug': 
        data_out=data_out.swapaxes(1, 2)
        data_out , labels_out = HHTaug_multi(data_out,labels_out)
        data_out=data_out.swapaxes(1, 2)

    # ==========================================
    # 9 model-based augmentation types:
    #   CNN-base / CNN-gan / CNN-vae
    #   CNN_Trans-base / CNN_Trans-gan / CNN_Trans-vae
    #   CNN_LSTM-base / CNN_LSTM-gan / CNN_LSTM-vae
    # ==========================================
    if aug_method in MODEL_BASED_AUGS:
        if info is not None:
            arch, loss_type = aug_method.rsplit('-', 1)
            aug_model_path = config.get('data_path', {}).get('aug_model', 'aug_models/')
            os.makedirs(aug_model_path, exist_ok=True)
            model_file = os.path.join(aug_model_path,
                                      f"kf{info[0]}set{info[1]}sub{info[2]}-{aug_method}.pth")

            # Auto-train if model does not exist
            if not os.path.exists(model_file):
                # data is (N, T, Nc), convert to (N, Nc, T) for training
                train_data_nc = data.swapaxes(1, 2)
                train_aug_generator(train_data_nc, labels, arch, loss_type, info, model_file)

            generator = torch.load(model_file, map_location=info[3], weights_only=False)
            generator.eval()
            # data is (N, T, Nc), convert to (N, 1, Nc, T) for generator
            data_tensor = torch.tensor(data.swapaxes(1, 2), dtype=torch.float).unsqueeze(1).to(info[3])
            with torch.no_grad():
                _, _, _, _, gen_data = generator(data_tensor)
            gen_data = gen_data.squeeze(1).cpu().detach().numpy()  # (N, Nc, T)
            gen_data = gen_data.swapaxes(1, 2)  # (N, T, Nc) - same format as data_out
            data_out = np.concatenate([data_out, gen_data], axis=0)
            labels_out = np.append(labels_out, labels)
        # else: info is None (batch-level call), skip model-based augmentation

    # Final output data layout: raw, mult_add, mult_reduce, noise, neg, freq1, freq2 (each 144 samples)
    data_out = data_out.swapaxes(1, 2)
    return data_out, labels_out


def data_noise_f(data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    noise_mod_val = 5  # previously 2
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
    mult_mod = 0.05
    # print("mult mod: {}".format(mult_mod))
    for i in range(len(labels)):
        if labels[i] >= 0:
            # print(data[i])
            data_t = data[i] * (1 + mult_mod)
            new_data.append(data_t)
            new_labels.append(labels[i])

    # for i in range(len(labels)):
    #     if labels[i] >= 0:
    #         data_t = data[i] * (1 - mult_mod)
    #         new_data.append(data_t)
    #         new_labels.append(labels[i])

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

    # for i in range(len(labels)):
    #     if labels[i] >= 0:
    #         high_shift = freq_shift(data[i], freq_mod, num_channels=n_channels)
    #         new_data.append(high_shift)
    #         new_labels.append(labels[i])

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

def DWTaug_multi(EEGData, EEGLabel):
    """
    Multi-class DWT-based data augmentation.
    Augmentation is performed independently within each class.
    """
    EEGLabel=EEGLabel.flatten()

    X_aug_all = [EEGData]
    y_aug_all = [EEGLabel]

    classes = np.unique(EEGLabel)

    for c in classes:
        mask = EEGLabel == c
        Xc = EEGData[mask]
        yc = EEGLabel[mask]

        # Reverse order as paired samples
        Xc_R = Xc[::-1]

        Xc_aug, yc_aug = DWTAug(Xc, Xc_R, yc, yc)

        X_aug_all.append(Xc_aug)
        y_aug_all.append(yc_aug)

    X_out = np.concatenate(X_aug_all, axis=0)
    y_out = np.concatenate(y_aug_all, axis=0)

    return X_out, y_out
def HHTaug_multi(EEGData, EEGLabel):
    """
    Multi-class DWT-based data augmentation.
    Augmentation is performed independently within each class.
    """
    EEGLabel=EEGLabel.flatten()

    X_aug_all = [EEGData]
    y_aug_all = [EEGLabel]

    classes = np.unique(EEGLabel)

    for c in classes:
        mask = EEGLabel == c
        Xc = EEGData[mask]
        yc = EEGLabel[mask]

        Xc_R = Xc[::-1]

        Xc_aug, yc_aug = HHTAug(Xc, Xc_R, yc)

        X_aug_all.append(Xc_aug)
        y_aug_all.append(yc_aug)

    X_out = np.concatenate(X_aug_all, axis=0)
    y_out = np.concatenate(y_aug_all, axis=0)

    return X_out, y_out
def DWTAug(Xs, Xt, ys, yt):
    # Euclidean alignment
    # if align:
    #     Xs = EA(Xs)
    #     X_tar_t = EA(X_tar_t)

    # Define wavelet function (optional)
    wavename = 'db4'

    # Signal decomposition
    ScA, ScD = pywt.dwt(Xs, wavename)
    TcA, TcD = pywt.dwt(Xt, wavename)

    # Cross-subject sub-signal reassembling and time domain reconstruction
    Xs_aug = pywt.idwt(ScA, TcD, wavename, 'smooth')  # Src approximated component + Tar detailed component
    Xt_aug = pywt.idwt(TcA, ScD, wavename, 'smooth')  # Tar approximated component + Src detailed component

    # Expand the training set: Final = [Original Xs & Augmented Xs & Augmented Xt]
    X_aug = np.concatenate((Xt_aug[:, :, :Xs.shape[-1]], Xs_aug[:, :, :Xs.shape[-1]]), axis=0)
    y_aug = np.concatenate((ys, yt), axis=0)  # Three parts have the same category

    return X_aug, y_aug

def HHTAug(Xs,Xt,ys):
    Xs_aug =[]
    Xt_aug = []
    for s in range(Xt.shape[0]):  # X_tar_t.shape[0]
        chns = []
        chnt = []
        for chn in range(Xt.shape[1]):
            # Signal decomposition
            imfs_tar = HHTFilter(Xt[s][chn])
            imfs_src = HHTFilter(Xs[s][chn])
            components_most_src = list(range(0, len(imfs_src) - 1, 1))
            components_less_src = [len(imfs_src) - 1]
            components_most_tar = list(range(0, len(imfs_tar) - 1, 1))
            components_less_tar = [len(imfs_tar) - 1]
            # Cross-subject sub-signal reassembling and reconstruction
            chns_aug = np.sum(imfs_src[components_most_src] + imfs_tar[components_less_tar], axis=0)
            chnt_aug = np.sum(imfs_tar[components_most_tar] + imfs_src[components_less_src], axis=0)
            chns.append(chns_aug)
            chnt.append(chnt_aug)
        chns = np.array(chns)
        chnt = np.array(chnt)
        Xs_aug.append(chns)
        Xt_aug.append(chnt)
    Xs_aug=np.array(Xs_aug)
    Xt_aug=np.array(Xt_aug)
    # Expand the training set: Final = [Original Xs & Augmented Xs & Augmented Xt]
    X = np.concatenate((Xt_aug[:, :, :Xs.shape[-1]], Xs_aug[:, :, :Xs.shape[-1]]), axis=0)
    y = np.concatenate((ys, ys), axis=0)  # Three parts have the same category
    return X,y
def nextpow2(x):
    return int(np.ceil(np.log2(np.abs(x))))


def HHTAnalysis(eegRaw, fs):
    # EMD decomposition
    decomposer = EMD(eegRaw)
    # Get IMF components after EMD
    imfs = decomposer.decompose()
    # Number of components after decomposition
    n_components = imfs.shape[0]
    # Define figure: raw data and each component
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
    # Plot instantaneous frequency; multiply by fs to convert normalized freq to Hz
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
    # Get IMF components
    imfs = decomposer.decompose()
    return imfs

def dct_transform(train_x, train_y):
    llist = [i for i in range(train_y.shape[0]) if train_y[i] == 0]
    rlist = [i for i in range(train_y.shape[0]) if train_y[i] == 1]
    Xl = train_x[llist, :, :]
    Xr = train_x[rlist, :, :]
    yl = train_y[llist]
    yr = train_y[rlist]
    Xld = dct(Xl, 1)
    Xrd = dct(Xr, 1)
    tlist = [0, int(train_x.shape[-1] / 10), int(train_x.shape[-1] / 10 * 2), int(train_x.shape[-1] / 10 * 3),
             int(train_x.shape[-1] / 10 * 4), int(train_x.shape[-1] / 10 * 5), int(train_x.shape[-1] / 10 * 6),
             int(train_x.shape[-1] / 10 * 7), int(train_x.shape[-1] / 10 * 8), int(train_x.shape[-1] / 10 * 9),
             train_x.shape[-1]]
    # tlist = [0, int(train_x.shape[-1]/5), int(train_x.shape[-1]/5*2), int(train_x.shape[-1]/5*3),
    #          int(train_x.shape[-1]/5*4), train_x.shape[-1]]
    # Concatenate segments
    allXld = []
    allXrd = []
    for k in range(len(llist)):
        random.seed(k)
        nums = random.sample(range(0, Xl.shape[0]), 11)  # sample N elements
        dllist = []
        drlist = []
        for i in range(1, len(nums)):
            dllist.append(Xld[nums[i], :, tlist[i - 1]:tlist[i]])
            drlist.append(Xrd[nums[i], :, tlist[i - 1]:tlist[i]])
        dllist = np.concatenate(dllist, axis=-1)
        drlist = np.concatenate(drlist, axis=-1)
        allXld.append(dllist)
        allXrd.append(drlist)
    allX = allXld + allXrd
    Xda = np.array(allX)
    X_aug = idct(Xda, 1)
    y_aug = np.concatenate((yl, yr), axis=0)
    return X_aug, y_aug   
        

def hs_transform(X, y):
    """
    Consider correspondence between original and augmented samples.
    Parameters
    ----------
    X: input original EEG signals
    y: the corresponding labels

    Returns
    X_la: left aug samples;
    X_ra: right aug samples;
    y_la, y_ra: the corresponding labels
    -------
    """
    seed = 42
    
    left_mat = [0,1,2]
    right_mat = [3,4]
    middle_mat = [5,6,7]
    random.seed(seed)
    np.random.seed(seed)
    num_samples, num_channels, num_timesamples = X.shape
    llist = [i for i in range(len(y)) if y[i] == 0]
    rlist = [i for i in range(len(y)) if y[i] == 1]
    Xl = X[llist, :, :]
    Xr = X[rlist, :, :]
    Xl_left = Xl[:, left_mat, :]
    Xl_right = Xl[:, right_mat, :]
    Xl_middle = Xl[:, middle_mat, :]
    Xr_left = Xr[:, left_mat, :]
    Xr_right = Xr[:, right_mat, :]
    Xr_middle = Xr[:, middle_mat, :]
    # print(Xl_left.shape, Xl_right.shape, Xl_middle.shape, Xr_left.shape, Xr_right.shape, Xr_middle.shape)
    # input('')
    llen = list(range(0, len(llist)))
    rlen = list(range(0, len(rlist)))
    transformedL2L = np.zeros((len(llist), num_channels, num_timesamples))
    transformedR2R = np.zeros((len(rlist), num_channels, num_timesamples))
    clist = left_mat + middle_mat + right_mat
    real_list = [clist.index(h) for h in range(0, num_channels)]
    for i in range(len(llist)):
        kl = random.choice([ele for ele in llen if ele != i])
        L2L = np.concatenate((Xl_left[kl, :, :], Xl_middle[i, :, :], Xl_right[i, :, :]), axis=0)  # concat left class -> 0
        L2L = np.take(L2L, real_list, axis=-2)  # reorder channel dimension
        transformedL2L[i, :, :] = L2L
    for i in range(len(rlist)):
        kr = random.choice([ele for ele in rlen if ele != i])
        R2R = np.concatenate((Xr_left[i, :, :], Xr_middle[i, :, :], Xr_right[kr, :, :]), axis=0)  # concat right class -> 1
        R2R = np.take(R2R, real_list, axis=-2)  # reorder channel dimension
        transformedR2R[i, :, :] = R2R
    aug_X = np.concatenate((transformedL2L, transformedR2R), axis=0)
    y_la = np.zeros((transformedL2L.shape[0]))
    y_ra = np.ones((transformedR2R.shape[0]))
    aug_y = np.concatenate((y_la, y_ra), axis=0)
    return aug_X, aug_y    
def freqSur(X):
    transform = FTSurrogate(
        probability=1.,  # defines the probability of actually modifying the input
    )
    X_tr, _ = transform.operation(torch.as_tensor(X).float(), None, 1, False)  # False when spatial info matters; phase perturbation in [0,1]
    return X_tr.numpy()


# class FTSurrogate(Transform):
#     """FT surrogate augmentation of a single EEG channel, as proposed in [1]_.

#     Parameters
#     ----------
#     probability: float
#         Float setting the probability of applying the operation.
#     phase_noise_magnitude : float | torch.Tensor, optional
#         Float between 0 and 1 setting the range over which the phase
#         perturbation is uniformly sampled:
#         ``[0, phase_noise_magnitude * 2 * pi]``. Defaults to 1.
#     channel_indep : bool, optional
#         Whether to sample phase perturbations independently for each channel or
#         not. It is advised to set it to False when spatial information is
#         important for the task, like in BCI. Default False.
#     random_state: int | numpy.random.Generator, optional
#         Seed to be used to instantiate numpy random number generator instance.
#         Used to decide whether or not to transform given the probability
#         argument. Defaults to None.

#     References
#     ----------
#     .. [1] Schwabedal, J. T., Snyder, J. C., Cakmak, A., Nemati, S., &
#        Clifford, G. D. (2018). Addressing Class Imbalance in Classification
#        Problems of Noisy Signals by using Fourier Transform Surrogates. arXiv
#        preprint arXiv:1806.08675.
#     """
#     operation = staticmethod(ft_surrogate)

#     def __init__(
#         self,
#         probability,
#         phase_noise_magnitude=1,
#         channel_indep=False,
#         random_state=None
#     ):
#         super().__init__(
#             probability=probability,
#             random_state=random_state
#         )
#         assert isinstance(phase_noise_magnitude, (float, int, torch.Tensor)), \
#             "phase_noise_magnitude should be a float."
#         assert 0 <= phase_noise_magnitude <= 1, \
#             "phase_noise_magnitude should be between 0 and 1."
#         assert isinstance(channel_indep, bool), (
#             "channel_indep is expected to be a boolean")
#         self.phase_noise_magnitude = phase_noise_magnitude
#         self.channel_indep = channel_indep

#     def get_augmentation_params(self, *batch):
#         """Return transform parameters.

#         Parameters
#         ----------
#         X : tensor.Tensor
#             The data.
#         y : tensor.Tensor
#             The labels.

#         Returns
#         -------
#         params : dict
#             Contains:

#             * phase_noise_magnitude : float
#                 The magnitude of the transformation.
#             * random_state : numpy.random.Generator
#                 The generator to use.
#         """
#         return {
#             "phase_noise_magnitude": self.phase_noise_magnitude,
#             "channel_indep": self.channel_indep,
#             "random_state": self.rng,
#         }


def data_preprocess(EEGData_Train, EEGData_Test):

    '''
    Parameters
    ----------
    EEGData_Train: EEG Training Dataset (Including Data and Labels)
    EEGData_Test: EEG Testing Dataset (Including Data and Labels)

    Returns: Preprocessed EEG DataLoader
    -------
    '''
    datasetid = config["train_param"]['datasets']
    bz = config["train_param"]['bz']
    Nm = config["model_param"]["Nm"]
    Nc = config[f"data_param{datasetid}"]["Nc"]  # number of target frequency

    '''Loading Training Data'''
    EEGData_Train, AUGData_Train,EEGLabel_Train,AUGLabel_Train = EEGData_Train[:]
    EEGData_Train = torch.tensor(EEGData_Train,dtype=torch.float)
    AUGData_Train = torch.tensor(AUGData_Train,dtype=torch.float)
    
    EEGLabel_Train = torch.tensor(EEGLabel_Train)
    AUGLabel_Train = torch.tensor(AUGLabel_Train)

    #EEGData_Train=EEGData_Train[:,:,[47, 53, 54, 55, 56, 57, 60, 61, 62],:]
    EEGData_Train = EEGData_Train.flatten(start_dim=1, end_dim=2)
    EEGData_Train = EEGData_Train.unsqueeze(1)
    AUGData_Train = AUGData_Train.flatten(start_dim=1, end_dim=2)
    AUGData_Train = AUGData_Train.unsqueeze(1)
    #X_train_noise, masks= add_noise(raw_eeg_train,noisetype=noisetype,noise_ratio=noise_ratio,snr_db=snr_db)
    print("EEGData_Train.shape", EEGData_Train.shape)
    print("AUGData_Train.shape", AUGData_Train.shape)
    # print("EEGLabel_Train.shape", EEGLabel_Train.shape)
    

    '''Loading Testing Data'''
    EEGData_Test, EEGLabel_Test = EEGData_Test[:]
    EEGData_Test = torch.tensor(EEGData_Test,dtype=torch.float)
    EEGLabel_Test = torch.tensor(EEGLabel_Test)
    #EEGData_Test=EEGData_Test[:,:,[47, 53, 54, 55, 56, 57, 60, 61, 62],:]
    EEGData_Test = EEGData_Test.flatten(start_dim=1, end_dim=2)
    EEGData_Test = EEGData_Test.unsqueeze(1)
    #X_test_noise, masks= add_noise(raw_eeg_test,noisetype=noisetype,noise_ratio=noise_ratio,snr_db=snr_db)
    print("EEGData_Test.shape", EEGData_Test.shape)
    # print("EEGnoise_Test.shape", X_test_noise.shape)
    #print("EEGLabel_Test.shape", EEGLabel_Test.shape)

    return EEGData_Train, AUGData_Train,EEGLabel_Train,AUGLabel_Train,EEGData_Test, EEGLabel_Test
