'''
=================================================
coding:utf-8
@Time:      2026/3/12 20:37
@File:      MMD.py
@Author:    Ziwei Wang
@Function:  Evaluation metric for generated data - Maximum Mean Discrepancy (MMD).
            Measures similarity between real and generated features; lower is more similar.
=================================================
'''

import numpy as np

def compute_mmd(real: np.ndarray, gen: np.ndarray, kernel='rbf', gamma=None) -> float:
    """MMD (Maximum Mean Discrepancy) with RBF kernel"""
    if gamma is None:
        gamma = 1.0 / real.shape[1]
    n_r, n_g = len(real), len(gen)
    K_rr = np.exp(-gamma * np.sum((real[:, None] - real[None, :]) ** 2, axis=2))
    K_gg = np.exp(-gamma * np.sum((gen[:, None] - gen[None, :]) ** 2, axis=2))
    K_rg = np.exp(-gamma * np.sum((real[:, None] - gen[None, :]) ** 2, axis=2))
    mmd = K_rr.sum() / (n_r * n_r) + K_gg.sum() / (n_g * n_g) - 2 * K_rg.mean()
    return float(np.sqrt(max(mmd, 0)))


