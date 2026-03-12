'''
=================================================
coding:utf-8
@File:      JaccardSimilarity.py
@Author:    Ziwei Wang
@Function:  生成数据评测指标 - Jaccard 相似度
            将连续特征二值化后计算集合相似度 |A∩B|/|A∪B|
=================================================
'''

import numpy as np
from typing import Optional, Literal


def _binarize(features: np.ndarray, method: str = "median", threshold: Optional[float] = None) -> np.ndarray:
    """将 (N, D) 特征二值化为 0/1。"""
    if method == "median":
        t = np.median(features, axis=0)
        return (features > t).astype(np.uint8)
    if method == "mean":
        t = np.mean(features, axis=0)
        return (features > t).astype(np.uint8)
    if method == "threshold" and threshold is not None:
        return (features > threshold).astype(np.uint8)
    return (features > 0).astype(np.uint8)


def compute_jaccard_similarity(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    binarize: Literal["median", "mean", "threshold"] = "median",
    threshold: Optional[float] = None,
    per_feature: bool = False,
) -> float:
    """
    在特征空间上计算真实集与生成集的 Jaccard 相似度。
    先将每维特征二值化（按 median/mean 或给定阈值），再在“样本×维度”的二元矩阵上
    定义集合：每行是一个样本的 D 维二值向量，视为集合的 D 个元素是否出现；
    或对每个特征维度分别算 0/1 比例后做 Jaccard。

    这里采用：对每个特征维度 d，统计 real 和 gen 上该维为 1 的比例 p_r, p_g，
    用 Jaccard = 交集/并集 的连续推广：对 [0,1] 比例用 J = min(p_r,p_g)/max(p_r,p_g) 或
    标准 J = |A∩B|/|A∪B|。对二值向量按样本聚合：A_d = {i: real[i,d]=1}, B_d = {i: gen[i,d]=1}，
    J_d = |A_d∩B_d|/|A_d∪B_d| 需要同一批样本；若 real 与 gen 样本不同，则用“分布级”的近似：
    每维上 J_d ≈ min(mean(real_d), mean(gen_d)) / max(mean(real_d), mean(gen_d))，再平均 over d。

    Parameters
    ----------
    real_features : np.ndarray, shape (N, D)
    gen_features : np.ndarray, shape (M, D)
    binarize : "median" | "mean" | "threshold"
        二值化方式
    threshold : float, optional
        binarize=="threshold" 时使用
    per_feature : bool
        True 时对每个特征维算 Jaccard 再平均；False 时将整表展平为二值集合再算一个 J

    Returns
    -------
    float
        Jaccard 相似度，取值 [0, 1]，越大越相似
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

    br = _binarize(real, method=binarize, threshold=threshold)
    bg = _binarize(gen, method=binarize, threshold=threshold)

    if per_feature:
        jaccards = []
        for j in range(d):
            p_r = np.mean(br[:, j])
            p_g = np.mean(bg[:, j])
            inter = min(p_r, p_g)
            union = max(p_r, p_g)
            if union < 1e-8:
                jaccards.append(1.0)
            else:
                jaccards.append(inter / union)
        return float(np.mean(jaccards))

    # 整表：每行是一个样本的二值向量，用“平均 Jaccard”近似
    # 定义：对每个维度 d，J_d = (sum(real_d) + sum(gen_d) 的交集近似) / 并集
    # 简化为：每维的 1 的比例的 Jaccard 平均
    p_r = np.mean(br, axis=0)
    p_g = np.mean(bg, axis=0)
    inter = np.minimum(p_r, p_g)
    union = np.maximum(p_r, p_g)
    union = np.where(union < 1e-8, 1.0, union)
    return float(np.mean(inter / union))


def compute_jaccard_similarity_1to1(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    binarize: Literal["median", "mean"] = "median",
) -> float:
    """
    当 real 与 gen 样本数相同且一一对应时，对每对样本的二值向量算 Jaccard，再取平均。
    Jaccard(a,b) = |a∩b| / |a∪b|，其中 a,b 为 D 维二值向量。

    Parameters
    ----------
    real_features : (N, D)
    gen_features : (N, D)
    binarize : "median" | "mean"
        在各自样本内或全局做二值化（这里用全局 median/mean 保证可比）

    Returns
    -------
    float
    """
    real = np.asarray(real_features, dtype=np.float64)
    gen = np.asarray(gen_features, dtype=np.float64)
    if real.shape != gen.shape:
        raise ValueError("real 与 gen 形状需一致")
    br = _binarize(real, method=binarize)
    bg = _binarize(gen, method=binarize)
    inter = np.sum(np.logical_and(br, bg), axis=1)
    union = np.sum(np.logical_or(br, bg), axis=1)
    union = np.where(union < 1e-8, 1.0, union)
    return float(np.mean(inter / union))


if __name__ == "__main__":
    # quick test
    N, M, D = 50, 50, 24
    real_features = np.random.randn(N, D).astype(np.float32) # replace with your features
    gen_features = np.random.randn(M, D).astype(np.float32) # replace with your features

    j = compute_jaccard_similarity(real_features, gen_features, binarize="median", per_feature=True)
    print(f"Jaccard similarity (per-feature): {j:.4f}")

    j1 = compute_jaccard_similarity_1to1(real_features, gen_features)
    print(f"Jaccard similarity (1-to-1 pairs): {j1:.4f}")
