# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/12/1 16:14
import scipy
from scipy import signal
import numpy as np
import torch
import pandas as pd
import math
import torch.nn as nn
class TRCA():
    def __init__(self, EEGDATA=None,n_components=1,is_ensemble=True):
        super(TRCA, self).__init__()
        self.EEGDATA = EEGDATA
        # print(EEGDATA.shape)
        if EEGDATA.ndim == 5:
            [self.Nf, self.Nm, self.Nc, self.T, _] = self.EEGDATA.shape
        else:
            raise ValueError(f'X.ndim must be 5 but got {EEGDATA.shape}')
        self.n_components = n_components
        self.trains = np.mean(self.EEGDATA, -1)
        self.W = torch.zeros((self.Nm, self.Nf, self.Nc))
        self.is_ensemble = is_ensemble

    def trca(self, X=None, labels=None):
        labels = labels
        for fb_i in range(self.Nm):
            for c in range(self.Nf):
                # 提取当前类别的所有试次数据
                if X.ndim == 4:
                    X_c = X[labels == c]
                    X_c = X_c[:, fb_i, :, :]
                else:
                    raise ValueError(f'X.ndim must be 4 but got {X.shape}')

                n_trials, n_channels, n_samples = X_c.shape
                # 初始化协方差矩阵
                S = np.zeros((n_channels, n_channels))  # 组内协方差
                Q = np.zeros((n_channels, n_channels))  # 组间协方差
                # 计算协方差矩阵
                for i in range(n_trials):
                    Xi = X_c[i].copy()
                    Xi -= Xi.mean(axis=1, keepdims=True)  # 中心化
                    Q += Xi @ Xi.T  # 更新组间协方差

                    # 计算组内协方差（避免重复计算）
                    for j in range(i + 1, n_trials):
                        Xj = X_c[j].copy()
                        Xj -= Xj.mean(axis=1, keepdims=True)
                        S += Xi @ Xj.T + Xj @ Xi.T
                Q += 1e-8 * np.eye(n_channels)
                try:
                    S_intra_inv = np.linalg.inv(Q)  # 计算类内协方差矩阵的逆
                except np.linalg.LinAlgError:
                    # 如果无法计算逆，使用伪逆
                    S_intra_inv = np.linalg.pinv(Q)
                # 计算广义特征值问题
                eigvals, eigvecs = np.linalg.eig(S_intra_inv @ S)
                sorted_indices = np.argsort(eigvals)[::-1]  # 从大到小排序
                W = eigvecs[:, sorted_indices[:self.n_components]].T
                self.W[fb_i, c, :] = torch.tensor(W, dtype=torch.float32)
        return self

    def test_trca(self, eeg, test_label):
        num_trials = eeg.shape[-1]
        fb_coefs = [math.pow(i, -1.25) + 0.25 for i in range(1, self.Nm + 1)]
        fb_coefs = np.array(fb_coefs)
        results = np.zeros((self.Nf, num_trials))
        # W=torch.tensor(torch.load(r'D:\gra_design\SSVEP-Retrain\WWW.pt'))
        for targ_i in range(self.Nf):
            if eeg.ndim == 5:
                test_tmp = eeg[targ_i, :, :, :, :]
            else:
                raise ValueError(f'X.ndim must be 5 but got {eeg.shape}')
            r = np.zeros((self.Nm, self.Nf, num_trials))
            for fb_i in range(self.Nm):
                if eeg.ndim == 5:
                    testdata = test_tmp[fb_i, :, :, :]
                for class_i in range(self.Nf):
                    if self.trains.ndim == 4:
                        traindata = self.trains[class_i, fb_i, :, :]
                    else:
                        traindata = self.trains[class_i, fb_i, :, class_i, :]
                    if not self.is_ensemble:
                        w = self.W[fb_i, class_i, :]
                    else:
                        w = self.W[fb_i, :, :]

                    for trial_i in range(num_trials):
                        testdata_w = np.matmul(w, testdata[:, :, trial_i])
                        traindata_w = np.matmul(w, traindata[:, :])
                        r_tmp = np.corrcoef(testdata_w.flatten(), traindata_w.flatten())
                        r[fb_i, class_i, trial_i] = r_tmp[0, 1]

            rho = np.einsum('j, jkl -> kl', fb_coefs, r)  # (num_targs, num_trials)
            tau = np.argmax(rho, axis=0)
            results[targ_i, :] = tau
        is_correct = (results == test_label)
        is_correct = np.array(is_correct).astype(int)
        test_acc = np.mean(is_correct)

        return test_acc





