'''
=================================================
coding:utf-8
@File:      JaccardSimilarity.py
@Author:    Ziwei Wang
@Function:  Evaluation metric for generated data - Jaccard similarity.
            Binarize continuous features then compute set similarity |A∩B|/|A∪B|.
=================================================
'''

import numpy as np
from typing import Optional, Literal


def _binarize(features: np.ndarray, method: str = "median", threshold: Optional[float] = None) -> np.ndarray:
    """Binarize (N, D) features to 0/1."""
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
    Compute Jaccard similarity between real and generated sets in feature space.
    First binarize each dimension (by median/mean or threshold), then define sets on the sample×dim binary matrix:
    each row is a D-dim binary vector per sample; or compute 0/1 ratio per dimension and then Jaccard.

    Here: for each dimension d, get proportion p_r, p_g of 1s in real and gen; use continuous extension
    J = min(p_r,p_g)/max(p_r,p_g), or standard J = |A∩B|/|A∪B|. For different sample sets we use
    distribution-level approx: J_d ≈ min(mean(real_d), mean(gen_d)) / max(mean(real_d), mean(gen_d)), then average over d.

    Parameters
    ----------
    real_features : np.ndarray, shape (N, D)
    gen_features : np.ndarray, shape (M, D)
    binarize : "median" | "mean" | "threshold"
        Binarization method.
    threshold : float, optional
        Used when binarize=="threshold".
    per_feature : bool
        If True, compute Jaccard per feature dimension then average; if False, one J over flattened table.

    Returns
    -------
    float
        Jaccard similarity in [0, 1]; higher is more similar.
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
        raise ValueError(f"Feature dimension mismatch: real {d} vs gen {d_g}")

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

    # Full table: each row is a sample binary vector; use "average Jaccard" approximation
    # Definition: per dimension d, J_d = (intersection approx of sum(real_d), sum(gen_d)) / union
    # Simplified: average of per-dimension Jaccard of 1-proportions
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
    When real and gen have the same size and are one-to-one paired, compute Jaccard per pair of binary vectors, then average.
    Jaccard(a,b) = |a∩b| / |a∪b|, where a,b are D-dim binary vectors.

    Parameters
    ----------
    real_features : (N, D)
    gen_features : (N, D)
    binarize : "median" | "mean"
        Binarize per sample or globally (here global median/mean for comparability).

    Returns
    -------
    float
    """
    real = np.asarray(real_features, dtype=np.float64)
    gen = np.asarray(gen_features, dtype=np.float64)
    if real.shape != gen.shape:
        raise ValueError("real and gen must have the same shape")
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
