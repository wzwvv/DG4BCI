'''
=================================================
coding:utf-8
@File:      FID.py
@Author:    Ziwei Wang
@Function:  生成数据评测指标 - Fréchet Inception Distance (FID) 
            衡量真实与生成特征的相似程度，越小越相似
=================================================
'''

import numpy as np
import scipy.linalg as scipy_linalg

def compute_fid(real_features: np.ndarray, gen_features: np.ndarray, eps=1e-6, normalize=True) -> float:
    """
    计算 FID (Fréchet Inception Distance)
    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*sqrt(Sigma_r @ Sigma_g))
    normalize=True 时先对特征做 z-score（用真实集统计量）
    """
    if normalize:
        real_features, gen_features = _normalize_features_for_fid(real_features, gen_features)
    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(gen_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False) + eps * np.eye(real_features.shape[1])
    sigma_g = np.cov(gen_features, rowvar=False) + eps * np.eye(gen_features.shape[1])
    diff = mu_r - mu_g

    try:
        covmean = scipy_linalg.sqrtm(sigma_r @ sigma_g)
        if np.iscomplexobj(covmean):
            covmean = np.real(covmean)
        fid = diff @ diff + np.trace(sigma_r) + np.trace(sigma_g) - 2 * np.trace(covmean)
    except Exception:
        fid = diff @ diff + np.trace(sigma_r) + np.trace(sigma_g)

    return float(np.clip(np.real(fid), 0, None))


def _normalize_features_for_fid(real_features: np.ndarray, gen_features: np.ndarray):
    """用真实集特征的均值和标准差对 real/gen 做 z-score，使 FID 数值更可解释、可比。"""
    mu = np.mean(real_features, axis=0)
    std = np.std(real_features, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    r = (real_features - mu) / std
    g = (gen_features - mu) / std
    return r, g