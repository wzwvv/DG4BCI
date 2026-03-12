'''
=================================================
coding:utf-8
@File:      EntropyStats.py
@Author:    Ziwei Wang
@Function:  生成数据评测指标 - 基于熵的统计量
            熵越高多样性越大；与真实分布熵的差异可衡量分布匹配
=================================================
'''

import numpy as np
from typing import Tuple, Optional


def _entropy_discrete(probs: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """离散分布的熵 H = -sum(p*log(p))，probs 沿 axis 和为 1。"""
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=axis)


def _entropy_continuous_histogram(x: np.ndarray, n_bins: int = 50, axis: int = -1) -> float:
    """用直方图估计一维或多维分布的熵（连续近似）。"""
    x = np.asarray(x).reshape(-1)
    hist, _ = np.histogram(x, bins=n_bins, density=True)
    hist = hist + 1e-8
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log(hist)))


def compute_entropy_stats(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    n_bins: int = 50,
) -> Tuple[float, float, float]:
    """
    基于熵的统计量（在特征空间上）。

    Parameters
    ----------
    real_features : np.ndarray, shape (N, D)
    gen_features : np.ndarray, shape (M, D)
    n_bins : int
        估计边际熵时的分箱数

    Returns
    -------
    real_entropy : float
        真实特征分布（每维直方图）的平均边际熵
    gen_entropy : float
        生成特征分布的平均边际熵
    entropy_diff : float
        |gen_entropy - real_entropy|，越小越接近
    """
    real = np.asarray(real_features, dtype=np.float64)
    gen = np.asarray(gen_features, dtype=np.float64)
    if real.ndim == 1:
        real = real.reshape(1, -1)
    if gen.ndim == 1:
        gen = gen.reshape(1, -1)
    n_r, d = real.shape
    n_g, d_g = gen.shape
    if d != d_g:
        raise ValueError(f"特征维度不一致: real {d} vs gen {d_g}")

    real_entropies = []
    gen_entropies = []
    for j in range(d):
        real_entropies.append(_entropy_continuous_histogram(real[:, j], n_bins=n_bins))
        gen_entropies.append(_entropy_continuous_histogram(gen[:, j], n_bins=n_bins))
    real_entropy = float(np.mean(real_entropies))
    gen_entropy = float(np.mean(gen_entropies))
    entropy_diff = float(np.abs(gen_entropy - real_entropy))
    return real_entropy, gen_entropy, entropy_diff


def compute_prediction_entropy(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    real_labels: np.ndarray,
    num_classes: int,
) -> Tuple[float, float]:
    """
    用真实数据训练分类器，在生成数据上预测，计算预测分布的熵。
    - 预测熵高：生成样本难以被分类（可能多样或噪声大）
    - 可与真实集上的预测熵对比

    Parameters
    ----------
    real_features : (N, D)
    gen_features : (M, D)
    real_labels : (N,) int
    num_classes : int

    Returns
    -------
    real_pred_entropy : float
        真实样本上的平均预测熵
    gen_pred_entropy : float
        生成样本上的平均预测熵
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("EntropyStats 需要 sklearn: pip install scikit-learn")

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_features)
    gen_scaled = scaler.transform(gen_features)
    clf = LogisticRegression(max_iter=500)
    clf.fit(real_scaled, real_labels)
    p_real = np.clip(clf.predict_proba(real_scaled), 1e-8, 1.0)
    p_gen = np.clip(clf.predict_proba(gen_scaled), 1e-8, 1.0)
    real_pred_entropy = float(np.mean(_entropy_discrete(p_real, axis=1)))
    gen_pred_entropy = float(np.mean(_entropy_discrete(p_gen, axis=1)))
    return real_pred_entropy, gen_pred_entropy


def compute_entropy_diversity(gen_features: np.ndarray, n_bins: int = 30) -> float:
    """
    仅对生成集：计算特征分布的（平均边际）熵，作为多样性指标。
    越大表示生成分布越分散。

    Parameters
    ----------
    gen_features : (M, D)
    n_bins : int

    Returns
    -------
    float
    """
    gen = np.asarray(gen_features).reshape(-1, gen_features.shape[-1])
    entropies = [_entropy_continuous_histogram(gen[:, j], n_bins=n_bins) for j in range(gen.shape[1])]
    return float(np.mean(entropies))


if __name__ == "__main__":
    # quick test
    N, M, D = 100, 80, 24
    real_features = np.random.randn(N, D).astype(np.float32)
    gen_features = np.random.randn(M, D).astype(np.float32)
    real_labels = np.random.randint(0, 2, size=N)

    real_h, gen_h, diff = compute_entropy_stats(real_features, gen_features, n_bins=30)
    print(f"Real entropy: {real_h:.4f}, Gen entropy: {gen_h:.4f}, |diff|: {diff:.4f}")

    real_pe, gen_pe = compute_prediction_entropy(
        real_features, gen_features, real_labels, num_classes=2
    )
    print(f"Prediction entropy — real: {real_pe:.4f}, gen: {gen_pe:.4f}")
