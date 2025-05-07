'''
=================================================
coding:utf-8
@Time:      2024/10/17 10:45
@File:      DWTAug-ML.py
@Author:    Ziwei Wang
@Function: Key code of Multilevel DWTAug
=================================================
'''


import numpy as np
import pywt

# Xs: The training data from each source subject
# X_tar_t: The training data from target subject
# X_tar_e: The test data of target subject
# X_tar_e is solely used for testing and remains unaugmented.
# Xs and X_tar_t have the same shape and the same category

# Euclidean alignment
if align:
    Xs = EA(Xs)
    X_tar_t = EA(X_tar_t)

# Define wavelet function (optional)
wavename = 'db4'

# Signal decomposition
Cs = pywt.wavedec(Xs, wavename, level=4)
ScA, ScD4, ScD3, ScD2, ScD1 = Cs[0], Cs[1], Cs[2], Cs[3], Cs[4]  # 4级
Ct = pywt.wavedec(X_tar_t, wavename, level=4)
TcA, TcD4, TcD3, TcD2, TcD1 = Ct[0], Ct[1], Ct[2], Ct[3], Ct[4]  # 4级

# Cross-subject sub-signal reassembling and time domain reconstruction
Xs_aug = pywt.waverec([ScA, TcD4, TcD3, TcD2, TcD1], wavename, 'smooth')  # Src approximated component + Tar detailed component
Xt_aug = pywt.waverec([TcA, ScD4, ScD3, ScD2, ScD1], wavename, 'smooth')  # Tar approximated component + Src detailed component

# Expand the training set: Final = [Original Xs & Augmented Xs & Augmented Xt]
Xs = np.concatenate((Xs, Xt_aug[:, :, :Xs.shape[-1]], Xs_aug[:, :, :Xs.shape[-1]]), axis=0)
ys = np.concatenate((ys, ys, ys), axis=0)  # Three parts have the same category