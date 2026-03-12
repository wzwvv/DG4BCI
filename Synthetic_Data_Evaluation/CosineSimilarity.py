'''
=================================================
coding:utf-8
@File:      CosineSimilarity.py
@Author:    Ziwei Wang
@Function:  生成数据评测指标 - 余弦相似度
            衡量真实与生成特征的相似程度，越大越相似
=================================================
'''

import numpy as np
from typing import Optional, Union


def compute_cosine_similarity(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    aggregate: str = "mean",
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """
    计算真实特征与生成特征之间的余弦相似度。

    Parameters
    ----------
    real_features : np.ndarray, shape (N, D)
        真实样本的特征向量
    gen_features : np.ndarray, shape (M, D)
        生成样本的特征向量
    aggregate : str
        - "mean": 先对每对 (real_i, gen_j) 或 (real_mean, gen_mean) 算余弦，再聚合
        - "mean_vectors": 只比较两个集合的均值向量，返回一个标量
        - "pairwise_mean": 对每个 real 找最近 gen 的余弦，再取平均（需 N<=M 或做双向）
    axis : optional
        若提供，在指定轴上归一化后再算（一般用默认 None）

    Returns
    -------
    float or np.ndarray
        标量或数组
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

    def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
        bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
        return np.sum(an * bn, axis=-1)

    if aggregate == "mean_vectors":
        mu_r = np.mean(real, axis=0)
        mu_g = np.mean(gen, axis=0)
        return float(_cos(mu_r[None], mu_g[None])[0])

    if aggregate == "pairwise_mean":
        # 每个 real 与所有 gen 的余弦，取最大再平均（或平均）
        cos_all = np.zeros(n_r)
        for i in range(n_r):
            c = _cos(gen, np.broadcast_to(real[i], (n_g, d)))
            cos_all[i] = np.mean(c)
        return float(np.mean(cos_all))

    # default: mean_vectors
    mu_r = np.mean(real, axis=0)
    mu_g = np.mean(gen, axis=0)
    return float(_cos(mu_r[None], mu_g[None])[0])


def compute_cosine_similarity_per_sample(
    real_features: np.ndarray,
    gen_features: np.ndarray,
) -> float:
    """
    要求 real 与 gen 样本数相同且一一对应，计算每对样本的余弦相似度后取平均。
    适用于条件生成中 real[i] 与 gen[i] 对应同一条件的情况。

    Parameters
    ----------
    real_features : (N, D)
    gen_features : (N, D)

    Returns
    -------
    float
        平均余弦相似度，取值 [-1, 1]，越大越相似
    """
    real = np.asarray(real_features, dtype=np.float64)
    gen = np.asarray(gen_features, dtype=np.float64)
    if real.shape != gen.shape:
        raise ValueError(f"形状需一致: real {real.shape} vs gen {gen.shape}")
    n = real.shape[0]
    rn = real / (np.linalg.norm(real, axis=1, keepdims=True) + 1e-8)
    gn = gen / (np.linalg.norm(gen, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(rn * gn, axis=1)))


if __name__ == "__main__":
    # quick test
    N, D = 50, 24
    real_features = np.random.randn(N, D).astype(np.float32)
    gen_features = np.random.randn(N, D).astype(np.float32)

    cs_mean = compute_cosine_similarity(real_features, gen_features, aggregate="mean_vectors")
    print(f"Cosine similarity (mean vectors): {cs_mean:.4f}")

    cs_pair = compute_cosine_similarity_per_sample(real_features, gen_features)
    print(f"Cosine similarity (per-sample, 1-to-1): {cs_pair:.4f}")
